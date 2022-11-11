import pandas as pd
from typing import Optional
import torch
from torch import nn
from transformers import BertForMaskedLM, BertTokenizer


def load_resctircted_vocab():
    vocab_df = pd.read_csv('/home/sxzou/concept_decoding/data/local_bert_vocab_nc_sci_ref.csv')
    keep_words = list(vocab_df.word.values)
    keep_ids = list(vocab_df.id.values)
    return keep_words, keep_ids


def get_restricted_vocab_for_bert(special_ids):
    # get the token ids in bart for our restricted vocab
    _, token_ids = load_resctircted_vocab()  # list of token ids that in our restricted vocab
    token_ids += special_ids  # special tokens
    token_ids = list(set(token_ids))
    token_ids.sort()

    # map the bart token id to our restricted vocab
    bert_token_id_2_restricted_token_id = {}
    for i, id in enumerate(token_ids):
        bert_token_id_2_restricted_token_id[id] = i
    return token_ids, bert_token_id_2_restricted_token_id


# 修改bert tokenizer
# encode: 先用tokenizer encode, 再把encode的id重新映射到我们的词表
# decode: 在调用`batch_decode`之前，先将我们的词表id映射回bert 的词表


class SmallBertForMaskedLM(BertForMaskedLM):
    '''
    Set the vocab to be our restricted vocab.
    '''
    def _get_resized_embeddings(self, old_embeddings: torch.nn.Embedding, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`): Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`): New number of tokens in the embedding matrix.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(self.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights
        special_ids = [102, 101, 103, 0, 100]
        token_ids_to_copy, _ = get_restricted_vocab_for_bert(special_ids)
        new_embeddings.weight.data = old_embeddings.weight.data[token_ids_to_copy, :]
        return new_embeddings

    def freeze_encoder(self):
        self._freeze_embeds()
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.get_encoder().parameters():
            param.requires_grad = True

    def _freeze_embeds(self):
        '''
        freeze the positional embedding parameters of the model
        adapted from finetune.py
        '''
        freeze_params(self.model.shared)
        d = self.model.decoder
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)


def freeze_params(model):
    '''
    Function that takes a model as input (or part of a model) and freezes the layers for faster training
    adapted from finetune.py
    '''
    for layer in model.parameters():
        layer.requires_grad = False





if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_ids = [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id,
                   tokenizer.unk_token_id]
    token_ids, bert_token_id_2_restricted_token_id = get_restricted_vocab_for_bert(special_ids)

    model = SmallBertForMaskedLM.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(new_num_tokens=len(token_ids))

    print(model)