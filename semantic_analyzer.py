import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from skimage import color
import plotly.express as px
from sentence_transformers import SentenceTransformer
import colorsys  # For color conversions

class SemanticAnalyzer:
    def __init__(self, strings, embedding_method='spacy', model_name=None, beta=0.5):
        """
        Initializes the SemanticAnalyzer with a list of strings and an embedding model.

        Parameters:
        - strings (list of str): The input strings.
        - embedding_method (str): The method to use for embeddings ('spacy' or 'sentence_transformer').
        - model_name (str): The name of the model to use.
            - For 'spacy', defaults to 'en_core_web_md' if None.
            - For 'sentence_transformer', defaults to 'clip-ViT-L-14' if None.
        - beta (float): The weighting factor for MaxPS in the diversity score (default is 0.5).
        """
        self.strings = strings
        self.embedding_method = embedding_method
        self.beta = beta
        self.embeddings = None
        self.similarity_matrix = None
        self.mps = None
        self.maxps = None
        self.ds = None

        # Load the embedding model
        if embedding_method == 'spacy':
            if model_name is None:
                model_name = 'en_core_web_md'
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"Model '{model_name}' not found. Downloading and installing...")
                from spacy.cli import download
                download(model_name)
                self.nlp = spacy.load(model_name)
        elif embedding_method == 'sentence_transformer':
            if model_name is None:
                model_name = 'clip-ViT-L-14'
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading model '{model_name}': {e}")
                raise
        else:
            raise ValueError("Unsupported embedding method. Use 'spacy' or 'sentence_transformer'.")

        # Compute embeddings and assign to self.embeddings
        self.embeddings = self.compute_embeddings()

        # Compute similarity matrix and assign to self.similarity_matrix
        self.similarity_matrix = self.compute_similarity()

        # Compute mps, maxps, and ds
        self.mps = self.get_mean_pairwise_similarity()
        self.maxps = self.get_max_pairwise_similarity()
        self.ds = self.get_diversity_score(beta=self.beta)

        # Print summary statistics
        print(self.__str__())

    def compute_embeddings(self):
        """
        Computes embeddings for the list of strings and returns them.

        Returns:
        - embeddings (ndarray): Array of embeddings.
        """
        if not self.strings:
            raise ValueError("The list of strings is empty.")

        if self.embedding_method == 'spacy':
            embeddings = np.array([self.nlp(text).vector for text in self.strings])
        elif self.embedding_method == 'sentence_transformer':
            embeddings = self.model.encode(self.strings)
        return embeddings

    def compute_similarity(self, metric='cosine'):
        """
        Computes the similarity matrix among the strings.

        Parameters:
        - metric (str): The similarity metric ('cosine' or 'euclidean').

        Returns:
        - similarity_matrix (ndarray): The similarity matrix.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings have not been computed.")

        if metric == 'cosine':
            similarity_matrix = cosine_similarity(self.embeddings)
        elif metric == 'euclidean':
            # Convert distances to similarities
            distances = euclidean_distances(self.embeddings)
            similarity_matrix = 1 / (1 + distances)
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

        return similarity_matrix

    def get_mean_pairwise_similarity(self):
        """
        Calculates and returns the Mean Pairwise Similarity (MPS).

        Returns:
        - mps (float): The mean of all pairwise similarities (excluding self-similarities).
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed.")

        # Exclude diagonal (self-similarities)
        sim_matrix = self.similarity_matrix.copy()
        np.fill_diagonal(sim_matrix, np.nan)
        # Compute mean of off-diagonal elements
        mps = np.nanmean(sim_matrix)
        return mps

    def get_max_pairwise_similarity(self):
        """
        Finds and returns the Maximum Pairwise Similarity (MaxPS).

        Returns:
        - maxps (float): The maximum similarity between any two different strings.
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed.")

        # Exclude diagonal
        sim_matrix = self.similarity_matrix.copy()
        np.fill_diagonal(sim_matrix, np.nan)
        maxps = np.nanmax(sim_matrix)
        return maxps

    def get_diversity_score(self, beta=0.5):
        """
        Calculates and returns the Diversity Score (DS).

        Parameters:
        - beta (float): The weighting factor for MaxPS (default is 0.5).

        Returns:
        - ds (float): The diversity score.
        """
        N = len(self.strings)
        mps = self.mps  # Mean Pairwise Similarity
        maxps = self.maxps  # Maximum Pairwise Similarity
        ds = N / (mps + beta * maxps)
        return ds

    def get_top_n_similar_pairs(self, n=5):
        """
        Returns a list of the top n pairs of strings with the highest similarity.

        Parameters:
        - n (int): The number of top pairs to return (default is 5).

        Returns:
        - top_pairs (list of tuples): Each tuple contains (similarity, string1, string2)
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed.")

        N = len(self.strings)
        sim_matrix = self.similarity_matrix.copy()
        # Exclude diagonal (self-similarities)
        np.fill_diagonal(sim_matrix, np.nan)
        # Get indices of upper triangle (to avoid duplicates)
        triu_indices = np.triu_indices(N, k=1)
        similarities = sim_matrix[triu_indices]
        # Get sorted indices in descending order
        sorted_indices = np.argsort(-similarities)
        # Get top n pairs
        top_n = min(n, len(sorted_indices))
        top_pairs = []
        for idx in range(top_n):
            i = triu_indices[0][sorted_indices[idx]]
            j = triu_indices[1][sorted_indices[idx]]
            sim = similarities[sorted_indices[idx]]
            str_i = self.strings[i]
            str_j = self.strings[j]
            top_pairs.append((sim, str_i, str_j))
        return top_pairs

    def plot_embeddings(self, perplexity=None, learning_rate=200, n_iter=1000, random_state=42, save_html=False, html_filename='embeddings_plot.html'):
        """
        Plots the embeddings using t-SNE and Plotly for interactivity, with colors assigned via MDS to reflect distances.
        The strings associated with points appear only when hovering over them.

        Parameters:
        - perplexity (int): The perplexity parameter for t-SNE (default is min(30, N//2)).
        - learning_rate (float): The learning rate for t-SNE (default is 200).
        - n_iter (int): The number of iterations for t-SNE (default is 1000).
        - random_state (int): The random seed for t-SNE (default is 42).
        - save_html (bool): Whether to save the plot as an HTML file (default is False).
        - html_filename (str): The filename to save the HTML plot (default is 'embeddings_plot.html').
        """
        
        if self.embeddings is None:
            raise ValueError("Embeddings have not been computed.")

        N = len(self.strings)
        max_perplexity = N - 1

        # Set default perplexity if None
        if perplexity is None:
            perplexity = min(30, max_perplexity // 2)
            print(f"No perplexity specified. Using default perplexity={perplexity}.")
        else:
            # Check if perplexity is higher than maximum allowed
            if perplexity >= max_perplexity:
                print(f"Specified perplexity={perplexity} is too high for N={N} data points.")
                perplexity = min(30, max_perplexity // 2)
                print(f"Using adjusted perplexity={perplexity}.")

        # Ensure perplexity is at least 1
        if perplexity < 1:
            print(f"Specified perplexity={perplexity} is too low. Setting perplexity=1.")
            perplexity = 1

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                    n_iter=n_iter, random_state=random_state)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        # Compute pairwise distances in 2D embedding space
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(embeddings_2d)

        # Apply MDS to map distances to 3D color space
        from sklearn.manifold import MDS
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state)
        embeddings_3d = mds.fit_transform(distance_matrix)

        # Normalize the 3D coordinates to [0, 1] for RGB colors
        x = embeddings_3d[:, 0]
        y = embeddings_3d[:, 1]
        z = embeddings_3d[:, 2]
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        z_norm = (z - z.min()) / (z.max() - z.min())

        # Stack normalized coordinates into RGB colors
        rgb_colors = np.stack((x_norm, y_norm, z_norm), axis=1)

        # Convert RGB to hex color codes
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r_ * 255), int(g_ * 255), int(b_ * 255)) for r_, g_, b_ in rgb_colors]

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'text': self.strings
        })

        # Create an interactive scatter plot without specifying the text parameter
        fig = px.scatter(
            df,
            x='x',
            y='y',
            hover_data={'text': True},
            title='Interactive t-SNE Plot of String Embeddings',
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
        )

        # Update marker colors directly
        fig.update_traces(marker=dict(color=hex_colors, size=10))
        fig.update_layout(
            width=800,
            height=600,
            hovermode='closest',
            showlegend=False  # Hide legend if colors are unique
        )

        # Show the plot
        fig.show()

        # Save the plot as an HTML file if requested
        if save_html:
            fig.write_html(html_filename)
            print(f"Interactive plot saved as {html_filename}.")

    def plot_embeddings_3d(self, perplexity=None, learning_rate=200, n_iter=1000, random_state=42, save_html=False, html_filename='embeddings_plot_3d.html'):
        """
        Plots the embeddings using t-SNE in 3D and creates an interactive 3D plot with Plotly, with colors assigned via MDS to reflect distances.
        The strings associated with points appear only when hovering over them.

        Parameters:
        - perplexity (int): The perplexity parameter for t-SNE (default is min(50, N//2)).
        - learning_rate (float): The learning rate for t-SNE (default is 200).
        - n_iter (int): The number of iterations for t-SNE (default is 1000).
        - random_state (int): The random seed for t-SNE (default is 42).
        - save_html (bool): Whether to save the plot as an HTML file (default is False).
        - html_filename (str): The filename to save the HTML plot (default is 'embeddings_plot_3d.html').   
        """
        if self.embeddings is None:
            raise ValueError("Embeddings have not been computed.")

        N = len(self.strings)
        max_perplexity = N - 1

        # Set default perplexity if None
        if perplexity is None:
            perplexity = min(50, max_perplexity // 2)
            print(f"No perplexity specified. Using default perplexity={perplexity}.")
        else:
            # Check if perplexity is higher than maximum allowed
            if perplexity >= max_perplexity:
                print(f"Specified perplexity={perplexity} is too high for N={N} data points.")
                perplexity = min(50, max_perplexity // 2)
                print(f"Using adjusted perplexity={perplexity}.")

        # Ensure perplexity is at least 1
        if perplexity < 1:
            print(f"Specified perplexity={perplexity} is too low. Setting perplexity=1.")
            perplexity = 1

        tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate,
                    n_iter=n_iter, random_state=random_state)
        embeddings_3d = tsne.fit_transform(self.embeddings)

        # Normalize the coordinates to [0, 1] for RGB colors
        x = embeddings_3d[:, 0]
        y = embeddings_3d[:, 1]
        z = embeddings_3d[:, 2]
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        z_norm = (z - z.min()) / (z.max() - z.min())

        # Stack normalized coordinates into RGB colors
        rgb_colors = np.stack((x_norm, y_norm, z_norm), axis=1)

        # Convert RGB to hex color codes
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r_ * 255), int(g_ * 255), int(b_ * 255)) for r_, g_, b_ in rgb_colors]

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'text': self.strings
        })

        # Create an interactive 3D scatter plot without specifying the text parameter
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            hover_data={'text': True},
            title='Interactive 3D t-SNE Plot of String Embeddings',
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'}
        )

        # Update marker colors directly
        fig.update_traces(marker=dict(color=hex_colors, size=5))
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=False  # Hide legend if colors are unique
        )

        # Show the plot
        fig.show()

        # Save the plot as an HTML file if requested
        if save_html:
            fig.write_html(html_filename)
            print(f"Interactive 3D plot saved as {html_filename}.")

    def __str__(self):
        """
        Returns a formatted string summarizing the statistics of the strings.
        """
        summary = []
        summary.append("\n-------------------------")
        summary.append("Semantic Analyzer Summary")
        summary.append("-------------------------")
        summary.append(f"Number of Strings: {len(self.strings)}")
        summary.append(f"Mean Pairwise Similarity (MPS): {self.mps:.4f}")
        summary.append(f"Maximum Pairwise Similarity (MaxPS): {self.maxps:.4f}")
        summary.append(f"Diversity Score (DS): {self.ds:.4f}\n")

        # Get top 5 similar pairs
        top_pairs = self.get_top_n_similar_pairs(n=5)
        summary.append("Top 5 Similar Pairs:")
        for sim, str1, str2 in top_pairs:
            summary.append(f"Similarity: {sim:.4f}")
            summary.append(f"Pair: ('{str1}', '{str2}')\n")

        summary.append("--------------------------------------------------\n")

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    strings = [
        "The apple is red and delicious.",
        "The sky is clear and blue today.",
        "Cherries are small, round, and often red.",
        "The car engine roared to life.",
        "Elderberries are used to make syrups and jams.",
        "The book was filled with thrilling adventures.",
        "Grapes can be eaten fresh or used to make wine.",
        "The software update improved performance.",
        "Kiwis have a tangy taste and are rich in vitamin C.",
        "The mountain peak was covered in snow."
    ]

    # Initialize the SemanticAnalyzer
    sa = SemanticAnalyzer(strings, embedding_method='spacy')

    # Plot the embeddings in 2D
    sa.plot_embeddings(perplexity=30, save_html=True)

    # Plot the embeddings in 3D
    sa.plot_embeddings_3d(perplexity=50)