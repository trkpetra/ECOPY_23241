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
import statsmodels.api as sm
from scipy.stats import t
from scipy.stats import f
import scipy.stats as stats

class LinearRegressionNP:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta = beta

    def get_params(self):
        beta_series = pd.Series(self.beta, name='Beta coefficients')
        return beta_series

    def get_pvalues(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        residuals = y - X @ beta
        residual_variance = (residuals @ residuals) / (n - k)
        standard_errors = np.sqrt(np.diagonal(residual_variance * np.linalg.inv(X.T @ X)))
        t_statistics = beta / standard_errors
        df = n - k
        pvalues = [2 * (1 - t.cdf(abs(t_stat), df)) for t_stat in t_statistics]
        pvalues = pd.Series(pvalues, name="P-values for the corresponding coefficients")
        return pvalues
    def get_wald_test_result(self, R):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = self.beta
        residuals = y - X @ beta
        r_matrix = np.array(R)
        r = r_matrix @ beta
        n = len(self.left_hand_side)
        m, k = r_matrix.shape
        sigma_squared = np.sum(residuals ** 2) / (n - k)
        H = r_matrix @ np.linalg.inv(X.T @ X) @ r_matrix.T
        wald = (r.T @ np.linalg.inv(H) @ r) / (m * sigma_squared)
        p_value = 1 - f.cdf(wald, dfn=m, dfd=n - k)
        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        y_pred = X @ beta
        ssr = np.sum((y_pred - np.mean(y)) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        centered_r_squared = ssr / sst
        adjusted_r_squared = 1 - (1 - centered_r_squared) * (n - 1) / (n - k)
        res = f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        return res

# Feasible GLS elvű

import numpy as np
import pandas as pd
from scipy.stats import t, f

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        self.Y = self.left_hand_side
        beta_ols = np.linalg.inv(self.X.T @ self.X) @ self.X.T @self.Y
        residuals = self.Y - self.X @ beta_ols
        log_residuals = np.log(residuals ** 2)
        w = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ log_residuals
        self.V_inv = np.diag(1 / np.sqrt(np.exp(self.X @ w)))
        beta_gls = np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ self.X.T @ self.V_inv @ self.Y
        self.beta_params = beta_gls

    def get_params(self):
        return pd.Series(self.beta_params, name='Beta coefficients')

    def get_pvalues(self):
        self.fit()
        dof = len(self.Y) - self.X.shape[1]
        residuals = self.Y - self.X @ self.beta_params
        residual_variance = (residuals @ residuals) / dof
        t_stat = self.beta_params / np.sqrt(np.diag(residual_variance*np.linalg.inv(self.X.T @ self.V_inv @ self.X)))
        res = pd.Series([min(value, 1 - value) * 2 for value in t.cdf(-np.abs(t_stat), df=dof)],
                             name='P-values for the corresponding coefficients')
        return res

    def get_wald_test_result(self, R):
        self.fit()
        r_matrix = np.array(R)
        r = r_matrix @ self.beta_params
        n = len(self.Y)
        m, k = r_matrix.shape
        residuals = self.Y - self.X @ self.beta_params
        residual_variance = (residuals @ residuals)/ (n - k)
        H = r_matrix @ np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ r_matrix.T
        wald = (r.T @ np.linalg.inv(H) @ r) / (m * residual_variance)
        p_value = 1 - f.cdf(wald, dfn=m, dfd=n - k)
        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        self.fit()
        sum_of_squares = self.Y.T @ self.V_inv @ self.Y
        residual_sum_of_squares = self.Y.T @ self.V_inv @ self.X @ np.linalg.inv(self.X.T @ self.V_inv @ self.X) @ self.X.T @ self.V_inv @ self.Y
        r_squared = 1 - (residual_sum_of_squares / sum_of_squares)
        adj_r_squared = 1 - (residual_sum_of_squares / (len(self.Y) - self.X.shape[1])) * (
                len(self.Y) - 1) / sum_of_squares
        return f"Centered R-squared: {r_squared:.3f}, Adjusted R-squared: {adj_r_squared:.3f}"


