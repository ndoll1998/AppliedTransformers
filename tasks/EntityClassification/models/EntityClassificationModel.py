from base import BaseModel

class EntityClassificationModel(BaseModel):
    """ Base Class for Entity Classification Models """

    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        entity_starts=None,
        entity_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """ Forward function for Entity Classification Models
            This needs to return the loss if labels are passed, 
            followed by entity logits
        """
        raise NotImplementedError()