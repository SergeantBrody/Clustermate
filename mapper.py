import pandas as pd
class Mapper:
    def __init__(self, clustered_data, Label_file):
            self.df = pd.read_csv(clustered_data)
            Labels = pd.read_csv(Label_file)
            cluster_Labels = {}
            for _, row in Labels.iterrows():
                asin = row['asin']
                label = row['Label']

                cluster = self.df[self.df['asin'] == asin]['cluster'].values
                if len(cluster) > 0 and cluster[0] != -1:
                    cluster_Labels[cluster[0]] = label
        
            self.df['Label'] = self.df['cluster'].map(cluster_Labels)

            self.df.loc[self.df['cluster'] == -1, 'Label'] = None
            self.df['Label'].fillna('Undecided', inplace=True)

            output_file = f'mapped_Labels_{clustered_data}.csv'
            self.df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

    def get_dataframe(self):

        return self.df