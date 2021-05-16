import torch
from applied import encoders
from applied import optimizers
from applied.tasks import relex
import matplotlib.pyplot as plt

# create encoder
encoder = encoders.BERT.from_pretrained("bert-base-uncased")
encoder.init_tokenizer_from_pretrained("bert-base-uncased")
# create model and optimizer
model = relex.models.MatchingTheBlanks(encoder=encoder, 
    num_labels=relex.datasets.GermYelp_Linking.num_labels())
optim = optimizers.AdamW(model.parameters(only_head=True), lr=1e-5, weight_decay=0.01)
# create dataset and prepare it for the model
dataset = relex.datasets.GermYelp_Linking(
    data_base_dir='../data', seq_length=128, batch_size=2)
# create trainer instance and train model
trainer = relex.Trainer(
    model=model, 
    dataset=dataset,
    optimizer=optim
).train(epochs=2)
# save metrics and model
trainer.metrics.save_table("../results/RelEx-Bert/metrics.table")
torch.save(model.state_dict(), "../results/RelEx-Bert/model.bin")
# plot metrics
fig = trainer.metrics.plot()
plt.show()