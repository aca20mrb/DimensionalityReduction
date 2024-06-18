import matplotlib.pyplot as plt
import seaborn as sns

# Define the provided data
methods = ['NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE',
           'NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE',
           'NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE', 'NR', 'PCA', 'K-PCA', 'RP', 't-SNE',
           'NR', 'PCA', 'K-PCA', 'RP', 't-SNE']

k_values = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]

silhouette_scores = [0.5817500491982808, 0.6145202036230449, 0.6311816485509132, 0.63702229234369, 0.8663443922996521,
                     0.45994823920518635, 0.5091683341538228, 0.6676968561248299, 0.5486653910107715, 0.6394619941711426,
                     0.38694104154427816, 0.44131690447171157, 0.6405637122084017, 0.5094121313084469, 0.6224074363708496,
                     0.34194697093163473, 0.4155809689613341, 0.6104076525254197, 0.4324129095571564, 0.49999886751174927,
                     0.32674451109112396, 0.43541709691019786, 0.5141499420843354, 0.4256769528561682, 0.4860450327396393,
                     0.5817500491982808, 0.5859517912768418, 0.5501836413191228, 0.652457043914747, 0.3566719591617584,
                     0.45994823920518635, 0.46613062910381436, 0.5588597971421404, 0.4150006008107933, 0.39439743757247925,
                     0.38694104154427816, 0.3942123888985152, 0.5444511314304752, 0.3328769024550932, 0.3803047239780426,
                     0.34194697093163473, 0.34961679380396193, 0.5304035344081323, 0.3090244036176294, 0.33752843737602234,
                     0.32674451109112396, 0.33444197671447523, 0.5443737337819022, 0.3266285224896285, 0.34088265895843506]

# Set seaborn style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))  # Consistent figure size

# Plotting results for varying K
unique_methods = list(set(methods))
palette = sns.color_palette("Blues", len(unique_methods))
color_mapping = {method: palette[i] for i, method in enumerate(unique_methods)}

for method in unique_methods:
    method_k_values = [k_values[i] for i in range(len(k_values)) if methods[i] == method]
    method_scores = [silhouette_scores[i] for i in range(len(silhouette_scores)) if methods[i] == method]
    plt.plot(method_k_values, method_scores, label=method, color=color_mapping[method])

plt.ylim(0, 1)  # Set y-axis limits
plt.title('Silhouette Score vs. K-Value')
plt.xlabel('K-Value')
plt.ylabel('Silhouette Score')
plt.legend(title='Method')

plt.show()
