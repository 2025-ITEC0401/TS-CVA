import matplotlib.pyplot as plt

def plot_loss_curves(name, total_loss_log, loss_log, val_loss_log, cutoff=1):
    # make everything less than cutoff value
    total_loss_log = [min(l, cutoff) for l in total_loss_log]
    loss_log = [min(l, cutoff) for l in loss_log]
    val_loss_log = [min(l, cutoff) for l in val_loss_log]


    # total loss, training loss, and validation loss in one plot
    plt.figure(figsize=(40, 20))
    plt.plot(total_loss_log, label='Total Loss', color='blue')
    plt.plot(loss_log, label='Training Loss', color='orange')
    plt.plot(val_loss_log, label='Validation Loss', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name}.png')
    plt.close()

if __name__ == "__main__":
    #BasicMotions__UEA_20251019_214218
    path = input("Enter the path to the loss files (without extension): ")
    total_loss_log = []
    loss_log = []
    val_loss_log = []

    with open(f'{path}/total_loss.txt', 'r') as f:
        total_loss_log = [float(line.strip()) for line in f.readlines()]
    with open(f'{path}/loss.txt', 'r') as f:
        loss_log = [float(line.strip()) for line in f.readlines()]
    with open(f'{path}/val_loss.txt', 'r') as f:
        val_loss_log = [float(line.strip()) for line in f.readlines()]

    plot_loss_curves(f'{path}/loss_curves', total_loss_log, loss_log, val_loss_log)