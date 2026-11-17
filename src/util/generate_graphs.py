# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz
#
# Declaration: The primary structure of this visualization script 
# was generated using artificial intelligence(ChatGPT) and subsequently modified 
# and extended for exact evaluation of experimental data.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# set global visual style for publication outputs
plt.style.use(['science', 'grid', 'no-latex'])
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# dataset categories
CATEGORY_LIST = (
    ['Vague'] * 60 +
    ['Negation'] * 60 +
    ['Multihop'] * 60 +
    ['False premise'] * 40 +
    ['Basic'] * 50 +
    ['Vague'] * 10 +
    ['Negation'] * 10 +
    ['Multihop'] * 10 +
    ['False premise'] * 10
)

CATEGORIES_ORDER = ['Basic', 'Vague', 'Negation', 'Multihop', 'False premise']

CATEGORY_DISPLAY_NAMES = {
    'Basic': 'Basic',
    'Vague': 'Vague',
    'Negation': 'Negation',
    'Multihop': 'Multi-hop',
    'False premise': 'False Premise'
}


# calculates mean and 95% confidence interval using bootstrap method
def bootstrap_ci(series, iterations=10000):

    data = series.dropna().values
    if len(data) == 0:
        return 0, 0, 0
        
    boot_means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(iterations)
    ]

    mean = np.mean(data)
    low = np.percentile(boot_means, 2.5)
    high = np.percentile(boot_means, 97.5)

    return mean, mean - low, high - mean


# aggregates statistical results for given architectures and metrics
def get_bootstrap_results(dfs, archs, metrics):

    means = {}
    err_low = {}
    err_high = {}

    for arch in archs:
        means[arch] = []
        err_low[arch] = []
        err_high[arch] = []

        for metric in metrics:
            mean, low, high = bootstrap_ci(dfs[arch][metric])
            means[arch].append(mean * 100)
            err_low[arch].append(low * 100)
            err_high[arch].append(high * 100)

    return means, err_low, err_high


# loads source result datasets and maps domain categories
def load_data():

    print("Loading data from CSV files...")

    files = {
        'Nomic Vector': '../../results/vector_nomic.csv',
        'BGE Vector': '../../results/vector_bge.csv',
        'BGE Lexical': '../../results/lexical_bge.csv',
        'BGE Hybrid': '../../results/hybrid_bge.csv',
        'BGE Hybrid+MQ': '../../results/multi_bge.csv',
        'BGE Full RAG': '../../results/reranker_bge.csv'
    }
    
    dataframes = {}
    for name, path in files.items():
        try:
            df = pd.read_csv(path, sep=None, engine='python')
            
            # dynamically handle mapped array length
            if len(df) == len(CATEGORY_LIST):
                df['category'] = CATEGORY_LIST
            else:
                df['category'] = CATEGORY_LIST[:len(df)]
                
            dataframes[name] = df
        except FileNotFoundError:
            print(f"WARNING: File {path} was not found.")
            
    return dataframes


def plot_embedding_comparison(dfs):

    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    labels = ['Context\nRecall', 'Context\nPrecision', 'Faithfulness', 'Answer\nRelevancy', 'Answer\nCorrectness']

    models = ['Nomic Vector', 'BGE Vector']
    model_labels = ['Nomic Embed', 'BGE-M3']

    means = {m: [] for m in models}
    err_low = {m: [] for m in models}
    err_high = {m: [] for m in models}

    # calculate statistics for embedding models
    for model in models:
        for metric in metrics:
            mean, low, high = bootstrap_ci(dfs[model][metric])
            means[model].append(mean * 100)
            err_low[model].append(low * 100)
            err_high[model].append(high * 100)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    width = 0.35

    # plot bars
    ax.bar(x - width / 2, means['Nomic Vector'], width, label=model_labels[0], color='#3498db', yerr=[err_low['Nomic Vector'], err_high['Nomic Vector']], capsize=5)
    ax.bar(x + width / 2, means['BGE Vector'], width, label=model_labels[1], color='#9b59b6', yerr=[err_low['BGE Vector'], err_high['BGE Vector']], capsize=5)

    ax.set_ylabel('Skóre (%)', fontweight='bold')
    ax.set_title('Porovnání embedding modelů', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = '../../text/graphs_text/embedding/embedding_comparison_bootstrap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_search_strategies_comparison(dfs):

    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    labels = ['Context\nRecall', 'Context\nPrecision', 'Faithfulness', 'Answer\nRelevancy', 'Answer\nCorrectness']

    archs = ['BGE Lexical', 'BGE Vector', 'BGE Hybrid']
    display_names = ['Lexikální\n(BM25)', 'Vektorové\n(sémantické)', 'Hybridní']
    colors = ['#95a5a6', '#3498db', '#9b59b6']

    means, err_low, err_high = get_bootstrap_results(dfs, archs, metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    # plot grouped bars for search strategies
    for idx, arch in enumerate(archs):
        offset = (idx - 1) * width
        ax.bar(x + offset, means[arch], width, label=display_names[idx], color=colors[idx], yerr=[err_low[arch], err_high[arch]], capsize=5)

    ax.set_ylabel('Skóre (%)', fontweight='bold')
    ax.set_title('Porovnání vyhledávacích strategií', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=3)

    plt.tight_layout()
    output_path = '../../text/graphs_text/search/graph_search_strategies_bootstrap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_metrics_by_question_type(dfs):

    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    metric_titles = {
        'context_recall': 'Context Recall', 'context_precision': 'Context Precision',
        'faithfulness': 'Faithfulness', 'answer_relevancy': 'Answer Relevancy',
        'answer_correctness': 'Answer Correctness'
    }

    archs = ['BGE Lexical', 'BGE Vector', 'BGE Hybrid']
    display_names = ['Lexikální', 'Vektorové', 'Hybridní']
    colors = ['#95a5a6', '#3498db', '#9b59b6'] 

    # generate separate category graphs for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(CATEGORIES_ORDER))
        width = 0.25

        for idx, arch in enumerate(archs):
            means = []
            err_low = []
            err_high = []

            for cat in CATEGORIES_ORDER:
                data_slice = dfs[arch][dfs[arch]['category'] == cat][metric]
                mean, low, high = bootstrap_ci(data_slice)

                means.append(mean * 100)
                err_low.append(low * 100)
                err_high.append(high * 100)

            offset = (idx - 1) * width
            ax.bar(x + offset, means, width, label=display_names[idx], color=colors[idx], yerr=[err_low, err_high], capsize=5)

        ax.set_ylabel('Skóre (%)', fontweight='bold')
        ax.set_title(f'{metric_titles[metric]} podle typu dotazu', pad=20, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_DISPLAY_NAMES[c] for c in CATEGORIES_ORDER], fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3)

        plt.tight_layout()
        output_path = f'../../text/graphs_text/search/category_{metric}_bootstrap.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_multiquery_effect(dfs):

    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    labels = ['Context\nRecall', 'Context\nPrecision', 'Faithfulness', 'Answer\nRelevancy', 'Answer\nCorrectness']

    archs = ['BGE Hybrid', 'BGE Hybrid+MQ']
    display_names = ['Hybridní', 'Hybridní + Multi-Query']
    colors = ['#3498db', '#9b59b6']

    means, err_low, err_high = get_bootstrap_results(dfs, archs, metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35

    # plot impact of multiquery expansion
    for idx, arch in enumerate(archs):
        offset = (idx - 0.5) * width
        ax.bar(x + offset, means[arch], width, label=display_names[idx], color=colors[idx], yerr=[err_low[arch], err_high[arch]], capsize=5)

    ax.set_ylabel('Skóre (%)', fontweight='bold')
    ax.set_title('Vliv Multi-Query Retrieval', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.tight_layout()
    output_path = '../../text/graphs_text/mq/graph_multiquery_effect_bootstrap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_multiquery_context_recall_by_category(dfs):

    metric = 'context_recall'
    archs = ['BGE Hybrid', 'BGE Hybrid+MQ']
    display_names = ['Hybridní', 'Hybridní + Multi-Query']
    colors = ['#3498db', '#9b59b6']

    means = {arch: [] for arch in archs}
    err_low = {arch: [] for arch in archs}
    err_high = {arch: [] for arch in archs}

    for arch in archs:
        for cat in CATEGORIES_ORDER:
            data_slice = dfs[arch][dfs[arch]['category'] == cat][metric]
            mean, low, high = bootstrap_ci(data_slice)

            means[arch].append(mean * 100)
            err_low[arch].append(low * 100)
            err_high[arch].append(high * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CATEGORIES_ORDER))
    width = 0.35

    for idx, arch in enumerate(archs):
        offset = (idx - 0.5) * width
        ax.bar(x + offset, means[arch], width, label=display_names[idx], color=colors[idx], yerr=[err_low[arch], err_high[arch]], capsize=5)

    ax.set_ylabel('Context Recall (%)', fontweight='bold')
    ax.set_title('Vliv Multi-Query Retrieval na Context Recall podle typu dotazu', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY_NAMES[c] for c in CATEGORIES_ORDER], fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.tight_layout()
    output_path = '../../text/graphs_text/mq/graph_multiquery_context_recall_by_category_bootstrap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reranker_architecture_comparison(dfs):

    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    labels = ['Context\nRecall', 'Context\nPrecision', 'Faithfulness', 'Answer\nRelevancy', 'Answer\nCorrectness']

    archs = ['BGE Hybrid', 'BGE Hybrid+MQ', 'BGE Full RAG']
    display_names = ['Hybridní', 'Hybridní + Multi-Query', 'Full RAG + Reranker']
    colors = ['#95a5a6', '#3498db', '#9b59b6']

    means, err_low, err_high = get_bootstrap_results(dfs, archs, metrics)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    width = 0.25

    # plot combined effect of multiquery and reranking
    for idx, arch in enumerate(archs):
        offset = (idx - 1) * width
        ax.bar(x + offset, means[arch], width, label=display_names[idx], color=colors[idx], yerr=[err_low[arch], err_high[arch]], capsize=5)

    ax.set_ylabel('Skóre (%)', fontweight='bold')
    ax.set_title('Vliv pokročilých komponent na kvalitu systému', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=3)

    plt.tight_layout()
    output_path = '../../text/graphs_text/reranker/reranker_architecture_comparison_bootstrap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reranker_metric_by_category(dfs, metric, metric_title, output_name):

    archs = ['BGE Hybrid', 'BGE Hybrid+MQ', 'BGE Full RAG']
    display_names = ['Hybridní', 'Hybridní + Multi-Query', 'Full RAG + Reranker']
    colors = ['#95a5a6', '#3498db', '#9b59b6']

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(CATEGORIES_ORDER))
    width = 0.25

    for idx, arch in enumerate(archs):
        means = []
        err_low = []
        err_high = []

        for cat in CATEGORIES_ORDER:
            data_slice = dfs[arch][dfs[arch]['category'] == cat][metric]
            mean, low, high = bootstrap_ci(data_slice)

            means.append(mean * 100)
            err_low.append(low * 100)
            err_high.append(high * 100)

        offset = (idx - 1) * width
        ax.bar(x + offset, means, width, label=display_names[idx], color=colors[idx], yerr=[err_low, err_high], capsize=5)

    ax.set_ylabel(f'{metric_title} (%)', fontweight='bold')
    ax.set_title(f'{metric_title} podle typu dotazu', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY_NAMES[c] for c in CATEGORIES_ORDER], fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=3)

    plt.tight_layout()
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_name}")
    plt.close()


def plot_reranker_context_precision_by_category(dfs):
    plot_reranker_metric_by_category(
        dfs=dfs,
        metric='context_precision',
        metric_title='Context Precision',
        output_name='../../text/graphs_text/reranker/reranker_context_precision_by_category_bootstrap.pdf'
    )


def plot_reranker_answer_correctness_by_category(dfs):
    plot_reranker_metric_by_category(
        dfs=dfs,
        metric='answer_correctness',
        metric_title='Answer Correctness',
        output_name='../../text/graphs_text/reranker/reranker_answer_correctness_by_category_bootstrap.pdf'
    )

if __name__ == "__main__":

    # initialize output directory structure for generated plots
    output_dirs = [
        '../../text/graphs_text/embedding',
        '../../text/graphs_text/search',
        '../../text/graphs_text/mq',
        '../../text/graphs_text/reranker'
    ]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
        
    dfs = load_data()
    
    # verify complete dataset availability before triggering generation
    if len(dfs) == 6:
        plot_embedding_comparison(dfs)
        plot_search_strategies_comparison(dfs)
        plot_metrics_by_question_type(dfs)
        
        plot_multiquery_context_recall_by_category(dfs)
        plot_multiquery_effect(dfs)

        plot_reranker_architecture_comparison(dfs)
        plot_reranker_context_precision_by_category(dfs)
        plot_reranker_answer_correctness_by_category(dfs)

        print("\nSuccess: All graphs were successfully generated and saved as PDF!")
    else:
        print("\nError: Missing source CSV files. Ensure all 6 files are present in the working directory.")