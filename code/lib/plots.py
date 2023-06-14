import matplotlib.pyplot as plt

def plot_loss_epoch(history):
    """
    plot evolution of training/validation loss and accuracy during training
    
    Args:
        history: training/validation loss and accuracy at each epoch
    """
    _,axes = plt.subplots(1,2,figsize=(12,6))

    axes[0].plot([i + 1 for i in range(len(history))], [history[i][0] for i in range(len(history))], color='r', label='training')
    axes[0].plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='b', label='validation')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')


    axes[1].plot([i + 1 for i in range(len(history))], [history[i][2] for i in range(len(history))], color='r', label='training')
    axes[1].plot([i + 1 for i in range(len(history))], [history[i][3] for i in range(len(history))], color='b', label='validation')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')

    axes[0].legend()
    axes[1].legend()
    plt.show()