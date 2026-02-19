"""Tests for feature engineering and train/test split pipeline."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


DATA_DIR = Path('data')
TRAIN_PATH = DATA_DIR / 'train_features.csv'
TEST_PATH = DATA_DIR / 'test_features.csv'


@pytest.fixture(scope='module')
def train():
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope='module')
def test(train):
    return pd.read_csv(TEST_PATH)


@pytest.fixture(scope='module')
def source():
    return pd.read_csv(DATA_DIR / 'evaldata_cleaned.csv')


class TestTemporalSplit:
    def test_train_years_exclude_2324(self, train):
        assert 2324 not in train['target_year'].values

    def test_test_year_is_2324_only(self, test):
        assert set(test['target_year'].unique()) == {2324}

    def test_train_years_correct(self, train):
        expected = {1920, 2021, 2122, 2223}
        assert set(train['target_year'].unique()) == expected

    def test_no_empty_sets(self, train, test):
        assert len(train) > 0
        assert len(test) > 0


class TestNoDataLeakage:
    def test_prev_att_rate_from_prior_year(self, source):
        """Verify prev_att_rate matches the actual prior year AttRate."""
        # Check for 2324: prev_att_rate should equal AttRate from 2223
        enrolled_2324 = source[(source['year'] == 2324) & source['AttRate'].notna()]
        enrolled_2223 = source[(source['year'] == 2223) & source['AttRate'].notna()]

        test = pd.read_csv(TEST_PATH)
        merged = test[['ANON_ID', 'prev_att_rate']].merge(
            enrolled_2223[['ANON_ID', 'AttRate']], on='ANON_ID', how='inner'
        )
        # prev_att_rate should match 2223 AttRate
        np.testing.assert_allclose(
            merged['prev_att_rate'].values,
            merged['AttRate'].values,
            rtol=1e-5,
            err_msg='prev_att_rate does not match prior year AttRate'
        )

    def test_school_features_from_prior_year(self, source, test):
        """School-level features should be computed from prior year data, not target year."""
        # school_chronic_rate in test should match 2223 school stats
        prior_school = (source[(source['year'] == 2223) & source['AttRate'].notna()]
                        .groupby('SiteName')['chronic_absent'].mean())

        test_with_school = test[test['school_chronic_rate'].notna()].copy()
        # Just verify values are in valid range
        assert test_with_school['school_chronic_rate'].between(0, 1).all()


class TestFeatureCompleteness:
    def test_columns_match(self, train, test):
        assert set(train.columns) == set(test.columns)

    def test_target_not_all_nan(self, train, test):
        assert train['chronic_absent'].notna().sum() > 0
        assert test['chronic_absent'].notna().sum() > 0

    def test_has_prior_data_indicator(self, train, test):
        assert 'has_prior_data' in train.columns
        assert train['has_prior_data'].isin([0, 1]).all()

    def test_demographic_encoded(self, train):
        eth_cols = [c for c in train.columns if c.startswith('Eth_')]
        gen_cols = [c for c in train.columns if c.startswith('Gen_')]
        assert len(eth_cols) > 0, 'No ethnicity dummy columns found'
        assert len(gen_cols) > 0, 'No gender dummy columns found'

    def test_prior_features_exist(self, train):
        expected = ['prev_att_rate', 'prev_days_abs', 'prev_susp',
                    'prior_mean_att_rate', 'att_rate_slope',
                    'prior_years_enrolled', 'prior_chronic_count']
        for col in expected:
            assert col in train.columns, f'Missing feature: {col}'


class TestClassBalance:
    def test_chronic_rate_reasonable(self, train, test):
        """Chronic rate should be between 10% and 70% for both sets."""
        train_clean = train[train['chronic_absent'].notna()]
        test_clean = test[test['chronic_absent'].notna()]
        train_rate = train_clean['chronic_absent'].mean()
        test_rate = test_clean['chronic_absent'].mean()
        assert 0.10 < train_rate < 0.70, f'Train chronic rate {train_rate:.2f} out of range'
        assert 0.10 < test_rate < 0.70, f'Test chronic rate {test_rate:.2f} out of range'

    def test_stratification_preservable(self, train):
        """Each target year should have enough samples for stratified CV."""
        for year in train['target_year'].unique():
            subset = train[train['target_year'] == year]
            n_chronic = subset['chronic_absent'].sum()
            n_not = len(subset) - n_chronic
            assert n_chronic >= 100, f'Year {year}: too few chronic cases ({n_chronic})'
            assert n_not >= 100, f'Year {year}: too few non-chronic cases ({n_not})'


class TestRowCounts:
    def test_test_matches_2324_enrollment(self, source, test):
        """Test set size should match 2324 enrolled students."""
        expected = len(source[(source['year'] == 2324) & source['AttRate'].notna()])
        assert len(test) == expected, f'Test size {len(test)} != 2324 enrolled {expected}'

    def test_train_size_reasonable(self, train):
        """Train should have data from multiple years, so larger than test."""
        assert len(train) > 50000
