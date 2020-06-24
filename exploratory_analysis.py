import numpy as np 
import config 
import datasets
import matplotlib.pyplot as plt 
import util

# Bar plot attach class (train and test)
def bar_plot_attach_class(y_train , y_test , path):
    # attach types and frequency
    unique_train , counts_train = np.unique(y_train, return_counts=True)
    unique_test , counts_test = np.unique(y_test, return_counts=True)
    # label locations 
    x = np.arange(len(unique_train))
    width= 0.35
    # create plot 
    fig , ax = plt.subplots(figsize=(9,6))
    rects1 = ax.bar(x-width/2 , counts_train , width , label='Train')
    rects2 = ax.bar(x+width/2 , counts_test , width , label='Test')
    # set title and label 
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_xlabel('Attach Class', fontweight='bold')
    ax.set_title('Network vector distribution in various NSL-KDD train and test dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_train)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(path)

# Bar plot protocol by attach class (train and test)
def bar_plot_protocol_attach(train , test , path):
    ## calculate counts protocol type
    # attach type 
    attach_type = np.unique(train[:,-1])
    attach_type = np.delete(attach_type , [1])
    # protocol type 
    protocol_type = np.unique(train[:,1])
    # frequency 
    train_counts = np.zeros((len(attach_type),len(protocol_type)),dtype=np.int32)
    test_counts = np.zeros((len(attach_type),len(protocol_type)),dtype=np.int32)
    for i in range(len(attach_type)):
        # get rows attach class equal attach type i 
        attach_train = train[:,:][train[:,-1] == attach_type[i]]
        attach_test = test[:,:][test[:,-1] == attach_type[i]]
        for j in range(len(protocol_type)):
            # get rows attach i and protocol j
            len_protocol_attach_train = attach_train[:,:][attach_train[:,1] == protocol_type[j]].shape[0]
            len_protocol_attach_test = attach_test[:,:][attach_test[:,1] == protocol_type[j]].shape[0]
            # add counts row attach i and  protocol j 
            train_counts[i][j] = len_protocol_attach_train
            test_counts[i][j] = len_protocol_attach_test
    
    counts=[train_counts , test_counts]
    title_axes = ['Train Dataset' , 'Test Dataset']
    protocol_type = [p.upper() for p in protocol_type]

    ## create plot
    fig , ax = plt.subplots(2,1,sharex=True,figsize=(10,11))
    for i in range(2):
        # label locations 
        x = np.arange(len(protocol_type))
        width= 0.2
        rects1 = ax[i].bar(x-width , counts[i][0,:] , width , label= attach_type[0])
        rects2 = ax[i].bar(x ,  counts[i][1,:] , width , label=attach_type[1])
        rects3 = ax[i].bar(x+width ,  counts[i][2,:] , width , label=attach_type[2])
        rects4 = ax[i].bar(x+2*width ,  counts[i][3,:] , width , label=attach_type[3])

        # set title and label 
        ax[i].set_ylabel('Count', fontweight='bold')
        ax[i].set_title(title_axes[i])
        ax[i].set_xticks(x+width/2)
        ax[i].set_xticklabels(protocol_type)
        ax[i].legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                if height > 0 :
                    ax[i].annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
    plt.xlabel('Protocol Type', fontweight='bold')
    #st=fig.suptitle('Protocol-Wise Attacks',fontsize=16)
    #plt.savefig(r'result\barplot_protocol_type.png',bbox_extra_artists=[st], bbox_inches='tight')
    plt.savefig(path ,bbox_inches='tight')

if __name__ == "__main__":
    
    ## step1: Load train and test data 
    X_train , y_train , _ = datasets.load_NSL_KDD(config.path_Train_NSL_KDD)
    X_test , y_test , _ = datasets.load_NSL_KDD(config.path_Test_NSL_KDD)

    ## step2: Barplot attach class and save to path (imbalance)
    path = r'result\barplot_attach_class_imbalance_dataset.png'
    bar_plot_attach_class(y_train,y_test , path)

    ## step3: Barplot attach class and save to path (balance)
    path = r'result\barplot_attach_class_balance_dataset.png'
    _,y_balance_train = util.resample_data(X_train , y_train)
    bar_plot_attach_class(y_balance_train,y_test , path)

    ## step4: Barplot protocol by attach class and save
    path = r'result\barplot_protocol_type.png'
    train = np.concatenate((X_train, y_train.reshape(-1,1)), 1)
    test = np.concatenate((X_test, y_test.reshape(-1,1)), 1)
    bar_plot_protocol_attach(train , test , path)


    
    
    



    


    