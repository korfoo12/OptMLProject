import matplotlib.pyplot as plt

def plot_loss_epoch(history):
    plt.plot([i + 1 for i in range(len(history))], [history[i][0] for i in range(len(history))], color='r', label='train loss')
    plt.plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='b', label='dev loss')
    plt.legend()
    plt.title('Training history')
    plt.show()