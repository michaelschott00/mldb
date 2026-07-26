import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from mldb.store import RunStore

store = RunStore("./data")
df = store.load_artifact_by_query(name="german", tags=["dataset"])

mlp_hparams = {"lr": 1e-3, "batch_size": 32, "hidden_dim": 32, "epochs": 3}
run_id = store.create_run(hparams=mlp_hparams, tags=["german", "mlp", "model"])
log_dir = store.open_directory(run_id)
writer = SummaryWriter(log_dir=log_dir)

target = "credit_risk"
features = df.drop(columns=[target, "uuid"])
categorical_columns = features.select_dtypes(include=["object", "str"]).columns
ids = df["uuid"]
X = pd.get_dummies(features, columns=categorical_columns)
y = df.loc[:, target] - 1  # XGBoost needs 0/1 labels.
ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
    ids, X, y, test_size=0.2, random_state=0
)

X_train = torch.tensor(X_train.astype(float).to_numpy(), dtype=torch.float32)  # type: ignore
X_test = torch.tensor(X_test.astype(float).to_numpy(), dtype=torch.float32)  # type: ignore
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)  # type: ignore
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)  # type: ignore


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = MLP(X_train.shape[1], mlp_hparams["hidden_dim"])
optimizer = torch.optim.Adam(model.parameters(), lr=mlp_hparams["lr"])
loss_fn = nn.BCEWithLogitsLoss()

n_samples = X_train.shape[0]
batch_size = mlp_hparams["batch_size"]

for epoch in range(mlp_hparams["epochs"]):
    model.train()
    perm = torch.randperm(n_samples)  # type: ignore
    for start in range(0, n_samples, batch_size):
        idx = perm[start : start + batch_size]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        train_loss = loss_fn(train_logits, y_train)
        train_acc = ((train_logits > 0).float() == y_train).float().mean()

        test_logits = model(X_test)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = ((test_logits > 0).float() == y_test).float().mean()

    writer.add_scalar("loss/train", train_loss.item(), epoch)
    writer.add_scalar("loss/test", test_loss.item(), epoch)
    writer.add_scalar("accuracy/train", train_acc.item(), epoch)
    writer.add_scalar("accuracy/test", test_acc.item(), epoch)

writer.close()

log_dir = store.list_directory_by_run(run_id)
print(f"Tensorboard logs at: {log_dir}")
