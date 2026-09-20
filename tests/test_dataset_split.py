import unittest

from training.prepare_memory_datasets import assign_splits


class DatasetSplitTests(unittest.TestCase):
    def test_groups_do_not_cross_splits(self):
        records = []
        for group in range(20):
            for sample in range(2):
                records.append({"metadata": {"group_id": f"road-{group}"}, "sample": sample})
        splits = assign_splits(records, 42)
        ownership = {}
        for split, values in splits.items():
            for value in values:
                group = value["metadata"]["group_id"]
                self.assertNotIn(group, ownership) if group not in ownership else self.assertEqual(split, ownership[group])
                ownership[group] = split
        self.assertEqual(len(records), sum(map(len, splits.values())))


if __name__ == "__main__":
    unittest.main()
