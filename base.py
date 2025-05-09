import torch.nn as nn

class AbstractModel(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super(AbstractModel, self).__init__()
        self.config = config

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def get_representation(self, batch):
        raise NotImplementedError('get representation method must be implemented.')
    def get_embeddings(self, items):
        raise NotImplementedError('get item_embeddings must be implemented.')

    def predict(self, batch, n_return_sequences=1):
        raise NotImplementedError('predict method must be implemented.')