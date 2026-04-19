import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec_AddInfo(SequentialRecommender):

    def __init__(self, config, dataset):
        super(SASRec_AddInfo, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.side_feature_size = config["side_feature_size"]
        self.side_position_embedding = nn.Embedding(self.max_seq_length, self.side_feature_size)
        self.side_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.side_feature_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.dense_layer = nn.Linear(self.hidden_size + self.side_feature_size, self.hidden_size)
        self.freeze_side_feature = config["freeze_side_feature"]
        self.side_embedding = nn.Embedding(self.n_items, self.side_feature_size, padding_idx=0)
        self.side_embedding.weight.requires_grad = not self.freeze_side_feature

        item_interaction_feature = dataset.item_feat[config["item_additional_feature"]].to(self.device)
        if len(item_interaction_feature.shape) < 2:
            item_interaction_feature = item_interaction_feature.unsqueeze(1)

        self.side_LayerNorm = nn.LayerNorm(self.side_feature_size, eps=self.layer_norm_eps)
        self.side_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self.side_embedding.weight.data.copy_(item_interaction_feature)
        del item_interaction_feature

        if config["load_pretrain"]:
            pretrained_path = config["checkpoint_dir"] + config["pretrained_name"]
            pretrained_parameter = torch.load(pretrained_path)["state_dict"]
            current_state_dict = self.state_dict()
            for key in list(pretrained_parameter.keys()):
                if key not in current_state_dict:
                    del pretrained_parameter[key]
            current_state_dict.update(pretrained_parameter)
            self.load_state_dict(current_state_dict)

        if config["freeze_Rec_Params"]:
            for p_name, param in self.named_parameters():
                param.requires_grad = False
                if any(k in p_name for k in ["side_position_embedding", "side_trm_encoder", "dense_layer", "side_LayerNorm", "side_dropout"]):
                    param.requires_grad = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        side_emb = self.side_embedding(item_seq) + self.side_position_embedding(position_ids)
        side_emb = self.side_LayerNorm(side_emb)
        side_emb = self.side_dropout(side_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output      = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        side_trm_output = self.side_trm_encoder(side_emb, extended_attention_mask, output_all_encoded_layers=True)

        output = self.dense_layer(torch.cat((trm_output[-1], side_trm_output[-1]), dim=-1))
        return self.gather_indexes(output, item_seq_len - 1)

    def calculate_loss(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output   = self.forward(item_seq, item_seq_len)
        pos_items    = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items    = interaction[self.NEG_ITEM_ID]
            pos_score    = torch.sum(seq_output * self.item_embedding(pos_items), dim=-1)
            neg_score    = torch.sum(seq_output * self.item_embedding(neg_items), dim=-1)
            return self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output   = self.forward(item_seq, item_seq_len)
        scores       = torch.mul(seq_output, self.item_embedding(interaction[self.ITEM_ID])).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq     = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output   = self.forward(item_seq, item_seq_len)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
