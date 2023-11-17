import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_scores(df,save_path):
    # Define color palette for the plots
    colors = sns.color_palette("husl", 3)

    # Plot r2
    sns.set(style='whitegrid')
    fig1 = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=df.index, y='r2', data=df, palette=colors)
    plt.xlabel('')
    plt.ylabel('r2')
    plt.title('R2 Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'r2_scores.png'))


    # Plot mape
    fig2 = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=df.index, y='mape', data=df, palette=colors)
    plt.xlabel('')
    plt.ylabel('MAPE')
    plt.title('MAPE Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'mape_scores.png'))



    # Plot rmse
    fig3 = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=df.index, y='rmse', data=df, color=colors[0], label='RMSE')
    plt.xlabel('')
    plt.ylabel('RMSE')
    plt.title('RMSE Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'rmse_scores.png'))

    # Plot mse
    fig4 = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=df.index, y='mse', data=df, color=colors[0], label='RMSE')
    plt.xlabel('')
    plt.ylabel('MSE')
    plt.title('MSE Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'mse_scores.png'))


    # Plot mae
    fig5 = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=df.index, y='mae', data=df, color=colors[1], label='MAE')
    plt.xlabel('')
    plt.ylabel('MAE')
    plt.title('MAE Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'mae_scores.png'))


def plot_actual_vs_predicted(y_true,y,t,save_path,data_option="DAILY"):
    #Figure 
    states = y_true.index
    fig = plt.figure(figsize=(20,20), constrained_layout=True)
    gs = fig.add_gridspec(nrows=7,ncols=5)

    if data_option=="DAILY":
        fig.suptitle("Day wise cases vs Predicted cases using HODMD",fontweight='bold',fontsize=30)
    if data_option=="CUM":
        fig.suptitle("Cumulative wise cases vs Predicted cases using HODMD",fontweight='bold',fontsize=30)

    for row_ in range(7):
        for col_ in range(5):
            idx = 5*row_ + col_
            state = states[idx]
            ax = fig.add_subplot(gs[row_,col_])
            ax.set_title(f"{state}")
            ax.set_xlabel("Days")
            ax.set_ylabel("# Cases")
            ax.plot(y_true.iloc[idx,:],'-r')
            ax.plot(t,y[idx,:],'-g')

    plt.legend(["True Cases", "Predicted Cases"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'statewise_prediction.png'))


def plot_error_box(y_true,y,t,save_path):
    np.abs((y_true.iloc[:,t]-y)).T.plot(kind="box")
    plt.xticks(rotation=90);
    plt.xlabel('')
    plt.ylabel('Absolute Error')
    plt.title('Distribution of  Absolute Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'box_plot_error.png'))


def plot_lambs_true(t,y,lambdas,save_path,st="RJ"):
    states = list(y.index)
    st_idx = states.index(st)

    fig = plt.figure(figsize=(14,10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2,ncols=1)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(t, y.iloc[st_idx,:])
    ax1.set_title(f"Case Counts {st}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Num cases")

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(t,np.abs(lambdas[st_idx,:]))
    ones = np.ones(len(t))
    ax2.plot(t,ones,"r:")
    ax2.set_title(f"Eigen Value ($\lambda$) for {st}")
    ax2.set_xlabel("t")
    ax2.set_ylabel("$\lambda$")

    plt.savefig(os.path.join(save_path,'lambdas.png'))











