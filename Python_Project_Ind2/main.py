import pandas as pd 

# Part A
def read_data(data):
    print(data.info())

file = 'groceries.csv'
data = pd.read_csv(file)

read_data(data)


# 2. Infer schema; print a data dictionary (columns, types, meaning).
print(data.info())
print(data.head())

print(data.describe())

dictionary = pd.DataFrame({
    'Column:': data.columns,
    'Data types': data.dtypes.values
    })
print(dictionary)

dictionary = pd.DataFrame({ 
    'Column:': data.columns,
    'Data types': data.dtypes.values, 
    'Meaning': [
        'Fruits',
        'Baked product',
        'Margarine',
        'Soup items',
    ]    })
print(dictionary)
# Basic Exploratory Data Analysis (EDA)
# Number of transactions
for i in data.columns:
    print(f'Number of transactions in {i}: {data[i].count()}')


# Number of unique products 
for j in data.columns:
    print(f'Number of unique products in {j}: {data[j].nunique()}')

# Basket size distribution (min/median/95th percentile)
basket_size = data.count(axis=1)
print(f'Min: {basket_size.min()}')
print(f"Median : {basket_size.median()}")
print(f"95th percentile: {basket_size.quantile(0.95)}")


# Top 20 products by frequency 

all_product_counts = {}

for k in data.columns: 
    product_counts = data[k].value_counts() 
    for product, count in product_counts.items(): 
        if product in all_product_counts: 
            all_product_counts[product] += count 
        else: 
            all_product_counts[product] = count

top_20 = pd.Series(all_product_counts).sort_values(ascending=False).head(20)
print(top_20)


# Part B 

# Standardize item names (lowercase, strip whitespace; optional: replace spaces with under-scores)
for b in data.columns:
    data[b] = data[b].str.lower()
    data[b] = data[b].str.strip()
    data[b] = data[b].str.replace(' ', '_')

# Remove empty/invalid items; drop baskets with fewer than 2 items (explain your choice)

new_basket_size = []
for check in basket_size.index: 
    if basket_size[check] >= 2:
        new_basket_size.append(check)
    else: 
        basket_size.drop(check, inplace=True)

print(new_basket_size)

#Create a canonical transactions table with columns: transaction id, items (list of strings), basket size. Persist as transactions clean.parquet (or CSV)

rows = []

for each in new_basket_size:
    transaction_id = each
    items = data.loc[each].dropna().tolist()
    basket_size = len(items)

    rows.append({
        'transaction_id': transaction_id,
        'items': items,
        'basket_size': basket_size
    })

canonical_data = pd.DataFrame(rows)
canonical_data.to_csv('canonical_clean.csv', index=False)
print(canonical_data)


# Part C 

#Create a product-level price map: assign each unique product a random unit price in a reasonable range (e.g., $0.50–$15.00) using a fixed random seed for reproducibility (e.g., np.random.default rng(42)). Save as product prices.csv with columns product, price
import numpy as np
final_price = [] 

all_products = [] 
for c in data.columns: 
    for product in data[c]: 
        if pd.notna(product): 
            all_products.append(product)

unique_products = []
for q in all_products: 
    if q not in unique_products: 
        unique_products.append(q)

random_var = np.random.default_rng(42) # random seed is included here
prices = random_var.uniform(0.5, 15.0, len(unique_products))

final_price = pd.DataFrame({
        'product': unique_products,
        'price': prices 
})

final_price.to_csv('final_price.csv', index=False)

    

#  Compute basket totals by summing unit prices for items in each transaction (assume quan-tity=1 unless specified)
basket_total = [] 

for it in canonical_data['items']:
    total = 0 
    for pr in it:
        if pr in final_price['product'].values:
            unit_price = final_price.loc[final_price['product'] == pr, 'price'].values[0]
            total += unit_price
        else: 
            total += 0 
    basket_total.append(total)

print(basket_total)

# Add a basket total column to the transactions table and export as transactions priced.csv
canonical_data['basket_total'] = basket_total
canonical_data.to_csv('transactions_priced.csv', index=False)


# Part D 
# Count pairs and triples of items that occur in the same basket. Define support count (number of baskets containing the itemset).
pair_count = {} 

for pair in canonical_data['items']: 
    pair = sorted(pair)
    for s in range(len(pair)): 
        for l in range(s+1, len(pair)): 
            itemset = (pair[s], pair[l]) 
            if itemset in pair_count: 
                pair_count[itemset] += 1 
            else: 
                pair_count[itemset] = 1

triple_count = {}
for triple in canonical_data['items']: 
    triple = sorted(triple) 
    for a in range(len(triple)): 
        for b in range(a+1, len(triple)): 
            for c in range(b+1, len(triple)): 
                itemset = (triple[a], triple[b], triple[c]) 
                if itemset in triple_count: 
                    triple_count[itemset] += 1 
                else: 
                    triple_count[itemset] = 1

# A more efficient version of this using combinations and counter 
from itertools import combinations
from collections import Counter 

pair_counter = Counter()
for items in canonical_data['items']:
    pair_counter.update(combinations(sorted(items), 2))

triple_counter = Counter()
for items in canonical_data['items']:
    triple_counter.update(combinations(sorted(items), 3))


# Make min count configurable (default 20). Return all pairs/triples meeting the threshold.
pair_count = dict(pair_counter)
triple_count = dict(triple_counter)

min_count = 20 

cleaned_pair_count = {}
for n in pair_count:
    if pair_count[n] >= min_count:
        cleaned_pair_count[n] = pair_count[n]

cleaned_triple_count = {}
for m in triple_count: 
    if triple_count[m] >= min_count: 
        cleaned_triple_count[m] = triple_count[m]
 
print(cleaned_pair_count)
print(cleaned_triple_count) 

# Compute top-k pairs and top-k triples by frequency (default k=10), with ties broken deter-ministically (alphabetical).
k = 10 
list_of_pairs = list((cleaned_pair_count.items()))
df_pairs = pd.DataFrame(list_of_pairs, columns=['Itemset', 'Count'])
df_pairs = df_pairs.sort_values(by=['Count', 'Itemset'], ascending=[False, True])
top_k_pairs = df_pairs.head(k)
print(top_k_pairs)

list_of_triples = list((cleaned_triple_count.items()))
df_triples = pd.DataFrame(list_of_triples, columns=['Itemset', 'Count'])
df_triples = df_triples.sort_values(by=['Count', 'Itemset'], ascending=[False, True])
top_k_triples = df_triples.head(k)
print(top_k_triples)


# Report both support count and support fraction for each itemset 
total_baskets = len(canonical_data) 

support_count_2 = top_k_pairs['Count']
support_fraction_2 = support_count_2 / total_baskets 
top_k_pairs['Support Fraction'] = support_fraction_2
print(top_k_pairs)

support_count_3 = top_k_triples['Count']
support_fraction_3 = support_count_3 / total_baskets 
top_k_triples['Support Fraction'] = support_fraction_3
print(top_k_triples)



# Part E 
# Bar chart of the top 15 individual items by frequency.
import matplotlib.pyplot as plt 
import numpy as np 

top_items = np.array(top_20.index[ :15])
frequency = np.array(top_20.values[ :15]) 

plt.bar(top_items, frequency, color='blue', width=0.5)
plt.title("Top 15 Individual Items by Frequency")
plt.xlabel("Item")
plt.ylabel("Frequency")
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.show()

# Bar chart of top-k pairs by support fraction.
import matplotlib.pyplot as plt 
import numpy as np 

plt.bar(top_k_pairs["Itemset"].astype(str), top_k_pairs["Support Fraction"], color='orange')
plt.title("Top-k pairs by support fraction")
plt.xlabel("Item Pair")
plt.ylabel("Support Fraction")
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.show()


# Heatmap of a co-occurrence matrix for the 25 most frequent items.
import seaborn as sns

dr = pd.read_csv("canonical_clean.csv")

dr["items"] = dr["items"].str.strip("[]").str.replace("'", "").str.split(", ")

item_counts = Counter([item for sublist in dr["items"] for item in sublist])
top_items = [item for item, _ in item_counts.most_common(25)]

co_matrix = pd.DataFrame(0, index=top_items, columns=top_items)

for items in dr["items"]:
    filtered = [i for i in items if i in top_items]
    for a, b in combinations(filtered, 2):
        co_matrix.loc[a, b] += 1
        co_matrix.loc[b, a] += 1  

plt.figure(figsize=(10, 8))
sns.heatmap(co_matrix, cmap="Blues")
plt.title("Heatmap of Co-occurrence Matrix - Top 25 Items")
plt.xlabel("Items")
plt.ylabel("Items")
plt.tight_layout()
plt.show()


# Distribution plot of basket size and basket total (histogram or ECDF).

dp = pd.read_csv('transactions_priced.csv')

plt.hist(dp["basket_size"], bins=20, color='green')

plt.title("Distribution plot of basket size")
plt.xlabel("Basket Size")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.hist(dp["basket_total"], bins=20, color='purple')

plt.title("Distribution plot of basket total")
plt.xlabel("Basket Total")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



