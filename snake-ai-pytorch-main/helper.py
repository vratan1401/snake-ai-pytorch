import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf()) # get current figure
    plt.clf() # clear figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1])) # adds text annotations to plot at (x,y) = (len(scores)-1, scores[-1])
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # scores/mean_scores[-1] represents last score/mean_score in the list
    plt.show(block=False) # displays plot w/o blocking the execution of the code
    plt.pause(.1) # pauses code execution for 0.1s allowing plot update
