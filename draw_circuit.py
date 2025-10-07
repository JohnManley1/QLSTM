# single cell: draw all 4 QLSTM gate circuits (ASCII + matplotlib)
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

# import your wrapper (adjust path if needed)
from QLSTM import QShallowRegressionLSTM

features = ["Close"]
            
# ---- instantiate (change num_sensors to len(features) in your real run) ----
Qmodel = QShallowRegressionLSTM(
    num_sensors=len(features),
    hidden_units=16,
    n_qubits=7,
    n_qlayers=1
)

# Access the inner QLSTM
qlstm = Qmodel.lstm

# Mapping from names to the QNode attributes in your QLSTM
qnode_map = {
    "forget": qlstm.qlayer_forget,
    "input": qlstm.qlayer_input,
    "update": qlstm.qlayer_update,
    "output": qlstm.qlayer_output,
}

# Helper to extract the TorchLayer's stored weights parameter robustly
def get_torchlayer_weights(torchlayer):
    # Common case: parameter is named "weights"
    if hasattr(torchlayer, "_parameters") and "weights" in torchlayer._parameters:
        return torchlayer._parameters["weights"].detach()
    # Fallback: take the first registered parameter
    params = list(torchlayer.parameters())
    if len(params) == 0:
        raise RuntimeError("No parameters found on the TorchLayer.")
    return params[0].detach()

# Prepare a dummy input with a batch dimension (shape (batch, n_qubits)).
# Important: VQC in your QLSTM expects a batch (so list(...) [0] yields the vector).
dummy_inputs = torch.randn(1, qlstm.n_qubits)  # batch=1, n_qubits

# Loop and draw
for name, qnode in qnode_map.items():
    torchlayer = qlstm.VQC[name]  # TorchLayer wrapper for this gate
    try:
        weights = get_torchlayer_weights(torchlayer)
    except Exception as e:
        print(f"Could not extract weights for gate {name}: {e}")
        continue

    print("\n" + "=" * 40)
    print(f"{name.upper()} gate (ASCII):\n")
    # qnode expects (inputs, weights)
    print(qml.draw(qnode)(dummy_inputs, weights))

    print(f"\n{name.upper()} gate (matplotlib):")
    fig_or_tuple = qml.draw_mpl(qnode)(dummy_inputs, weights)
    # qml.draw_mpl may return a Figure or (fig, ax); handle either case
    if isinstance(fig_or_tuple, tuple):
        fig = fig_or_tuple[0]
    else:
        fig = fig_or_tuple
    plt.show()
