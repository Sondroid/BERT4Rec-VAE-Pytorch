from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popular_items = self.items_by_popularity()

        negative_samples = {}
        print('Sampling negative items')
        for user in self.eval.keys():
            seen = set(self.eval[user])

            samples = []
            for item in popular_items:
                if len(samples) == self.sample_size:
                    break
                if item in seen:
                    continue
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in self.train.keys():
            popularity.update(self.train[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items
