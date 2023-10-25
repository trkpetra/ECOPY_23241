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

