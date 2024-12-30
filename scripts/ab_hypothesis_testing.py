import pandas as pd
from scipy.stats import f_oneway, ttest_ind


class ABHypothesisTesting:
    def __init__(self, data):
        self.data = data

    def test_risk_across_provinces(self):
        """
        Test if there are significant risk differences (Total Claims) across provinces.
        Null Hypothesis: There are no risk differences across provinces.
        """
        province_groups = [self.data[self.data['Province'] == p]['Total_Claim'] for p in self.data['Province'].unique()]
        result = f_oneway(*province_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No risk differences across provinces",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_risk_between_zipcodes(self):
        """
        Test if there are significant risk differences (Total Claims) between zip codes.
        Null Hypothesis: There are no risk differences between zip codes.
        """
        zipcode_groups = [self.data[self.data['Zipcode'] == z]['Total_Claim'] for z in self.data['Zipcode'].unique()]
        result = f_oneway(*zipcode_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No risk differences between zip codes",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_margin_difference_between_zipcodes(self):
        """
        Test if there are significant margin (profit) differences between zip codes.
        Null Hypothesis: There are no significant margin differences between zip codes.
        """
        self.data['Margin'] = self.data['Premium'] - self.data['Total_Claim']
        zipcode_groups = [self.data[self.data['Zipcode'] == z]['Margin'] for z in self.data['Zipcode'].unique()]
        result = f_oneway(*zipcode_groups)
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No significant margin differences between zip codes",
            "F-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }

    def test_risk_difference_gender(self):
        """
        Test if there are significant risk differences (Total Claims) between genders.
        Null Hypothesis: There are no significant risk differences between women and men.
        """
        male_group = self.data[self.data['Gender'] == 'Male']['Total_Claim']
        female_group = self.data[self.data['Gender'] == 'Female']['Total_Claim']
        result = ttest_ind(male_group, female_group, equal_var=False)
        return {
            "Test": "T-Test",
            "Null Hypothesis": "No significant risk differences between women and men",
            "T-Statistic": result.statistic,
            "p-Value": result.pvalue,
            "Reject Null": result.pvalue < 0.05
        }
