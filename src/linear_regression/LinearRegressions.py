from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import pathlib
import typing

#amazon_df = merged_df[merged_df['Symbol'] == 'AMZN']
#amazon_df = amazon_df.drop(columns=['Symbol'])

class LinearRegressionSM:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        y = self.left_hand_side
        model = sm.OLS(y, X).fit()
        self._model = model

    def get_params(self):
        beta_params = self._model.params
        beta_params.name = 'Beta coefficients'
        return beta_params

    def get_pvalues(self):
        pvalues_df = pd.Series(self._model.pvalues, name="P-values for the corresponding coefficients")
        return pvalues_df 

    def get_wald_test_result(self, restriction_matrix):
        wald_test = self._model.wald_test(restriction_matrix, scalar=True)
        fvalue = round(wald_test.fvalue, 2)
        pvalue = round(wald_test.pvalue, 3)
        result_string = f"F-value: {fvalue}, p-value: {pvalue}"
        return result_string
    def get_model_goodness_values(self):
        adjusted_r_squared = self._model.rsquared_adj
        aic = self._model.aic
        bic = self._model.bic

        result_string = f"Adjusted R-squared: {adjusted_r_squared:.3f}, Akaike IC: {aic:.2e}, Bayes IC: {bic:.2e}"
        return result_string


import numpy as np
from scipy.stats import t, f

class LinearRegressionNP:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        X = self.right_hand_side.copy()
        y = self.left_hand_side
        X.insert(0, 'Intercept', 1)


        XTX_inv = np.linalg.inv(X.T @ X)
        beta = XTX_inv @ X.T @ y
        self.beta = beta

    def get_params(self):
        beta_series = pd.Series(self.beta, name='Beta coefficients')
        return beta_series

    def get_pvalues(self):
        X = self.right_hand_side.copy()
        X.insert(0, 'Intercept', 1)
        y = self.left_hand_side
        n = len(y)
        p = len(X.columns)

        RSS = ((y - X @ self.beta) ** 2).sum()
        sigma_sq = RSS / (n - p)

        C_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diagonal(C_inv) * sigma_sq)

        t_stats = self.beta / se
        p_values = 2 * (1 - np.abs(t_stats))
        p_values = np.minimum(p_values, 2 - p_values)

        p_values_series = pd.Series(p_values, name='P-values for the corresponding coefficients')
        return p_values_series
    def get_wald_test_result(self, R):
        X = self.right_hand_side.copy()
        X.insert(0, 'Intercept', 1)
        y = self.left_hand_side
        n = len(y)
        p = len(X.columns)

        RSS = ((y - X @ self.beta) ** 2).sum()
        sigma_sq = RSS / (n - p)

        C_inv = np.linalg.inv(X.T @ X)

        R = np.array(R)
        m = R.shape[0]
        r = R @ self.beta
        M = R @ C_inv @ R.T
        F = (r.T @ np.linalg.inv(M) @ r) / m
        p_value = 1 - stats.f.cdf(F, m, n - p)

        result = f'Wald: {F:.3f}, p-value: {p_value:.3f}'
        return result

    def get_model_goodness_values(self, include_intercept=False):
        X = self.right_hand_side.copy()
        if include_intercept:
            X.insert(0, 'Intercept', 1)
        y = self.left_hand_side
        n = len(y)
        p = len(X.columns)

        RSS = ((y - X @ self.beta) ** 2).sum()
        TSS = ((y - y.mean()) ** 2).sum()
        R_squared = 1 - RSS / TSS
        adjusted_R_squared = 1 - (1 - R_squared) * ((n - 1) / (n - p - 1))

        result = f'Centered R-squared: {R_squared:.3f}, Adjusted R-squared: {adjusted_R_squared:.3f}'
        return result