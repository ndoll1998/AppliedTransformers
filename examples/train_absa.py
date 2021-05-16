import torch
from applied import encoders
from applied import optimizers
from applied.tasks import absa
import matplotlib.pyplot as plt

# create encoder
encoder = encoders.BERT.from_pretrained("bert-base-uncased")
encoder.init_tokenizer_from_pretrained("bert-base-uncased")
# create model and optimizer
model = absa.models.SentencePairClassifier(encoder=encoder, 
    num_labels=absa.datasets.SemEval2014Task4.num_labels())
optim = optimizers.AdamW(model.parameters(only_head=True), lr=1e-5, weight_decay=0.01)
# create dataset and prepare it for the model
dataset = absa.datasets.SemEval2014Task4(
    data_base_dir='../data', seq_length=128, batch_size=2)
# create trainer instance and train model
trainer = absa.Trainer(
    model=model, 
    dataset=dataset,
    optimizer=optim
).train(epochs=2)
# save metrics and model
trainer.metrics.save_table("../results/ABSA-Bert/metrics.table")
torch.save(model.state_dict(), "../results/ABSA-Bert/model.bin")
# plot metrics
fig = trainer.metrics.plot()
plt.show()