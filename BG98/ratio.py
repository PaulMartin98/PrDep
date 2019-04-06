# MATLAB-like plotting framework
import matplotlib.pyplot as plt
# Fundamental package for scientific computing
# import numpy as np
import math
# Easy-to-use data structures and data analysis tools
import pandas as pd
# Statistical data exploration / OLS: Ordinary Least Squares
from statsmodels.formula.api import ols


class ratioCore(object):
    """ratio calculation core class"""

    def __init__(self):
        """Define default
           core parameters"""
        # Filename for importing
        self.filename = None
        # Plot title
        self.title = None
        # Imported data to process
        self.data = pd.DataFrame()  # Empty dataframe
        # Forecasted data
        self.forecast = pd.DataFrame()  # Empty dataframe
        self.confinf = pd.DataFrame()  # Empty dataframe
        self.confsup = pd.DataFrame()  # Empty dataframe
        # Maximum number of rows and columns
        self.maxRows, self.maxColumns = None, None
        # Definitieve regression result
        self.result = pd.DataFrame()  # Empty dataframe
        # Columns name for result
        self.columnResult = ["ratio_tuple", "ratio", "r_square", "std_err",
                             "pvalue", "conf_inf", "conf_sup", "num_pt"]
        # Set color for MIS
        self.colorMis = {"0MIS":  "#66FFFF",  "1MIS": "#0070C0",  "3MIS": "#FF00FF",
                         "6MIS":  "#000000",  "9MIS": "#1D8F1D", "12MIS": "#FF0000",
                         "15MIS": "#66FFFF", "18MIS": "#0070C0", "21MIS": "#FF00FF",
                         "24MIS": "#000000", "30MIS": "#1D8F1D", "36MIS": "#FF0000"}

    def getData(self, csvFilename, start_offset=1, end_offset=1):
        """Import and process data"""
        # Set filename for importing
        self.filename = csvFilename
        # Set plot title
        self.title = csvFilename.split("\\")
        self.title = self.title[-1][:-4]
        # Import data from file
        self.data = pd.read_csv(self.filename, sep=';', decimal=",")
        # Get max number of rows and columns
        self.maxRows, self.maxColumns = self.data.shape
        # Set offset for data processing
        self.start_offset = start_offset
        self.end_offset = end_offset
        # Process data
        self.__dataProcessing()
        # Forecast data
        self.__forecastData()

    def calculateRatio(self, ratios):
        """Calculate ratio
           from ratios list"""
        self.__calculateRatio([self.__convertToMisTuple(ratio)
                               for ratio in ratios])

    def displayData(self):
        """Display data
           from import"""
        print(self.data)

    def displayResult(self):
        """Display result
           from regression"""
        print(self.result)

    def displayForecast(self):
        """Display result
           from regression"""
        print(self.forecast)

    def plotR2(self):
        """Plot RSquare"""
        ax = self.result.reset_index().plot.bar(x='index', y='r_square',
                                                rot=0, figsize=(20, 20),
                                                title=self.title)
        plt.show()

    def plotMis(self, mis):
        """Plot MIS
            mis: list of MIS string"""
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        for m in mis:
                self.data.plot(x='DATE', y=m, kind='line',
                               linestyle="-", marker='o',
                               grid=True, ax=ax, title=self.title,
                               linewidth=2, color=self.colorMis[m])
                self.forecast.plot(x='DATE', y=m, kind='line',
                                   linestyle="--", marker='o', grid=True,
                                   ax=ax, title=self.title, linewidth=1,
                                   color=self.colorMis[m])                
        leg = plt.legend(loc='best')
        plt.xticks(range(0, len(self.data["DATE"])), self.data["DATE"], rotation=45)
        plt.show()

    def plotConfMis(self, mis):
        """Plot Confident interval"""
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        for m in mis:
                self.data.plot(x='DATE', y=m, kind='line',
                               linestyle="-", marker='o',
                               grid=True, ax=ax, title=self.title,
                               linewidth=2, color=self.colorMis[m])
                ax.fill_between(self.data["DATE"], self.confinf[m],
                                self.confsup[m], color=self.colorMis[m])
        leg = plt.legend(loc='best')
        plt.xticks(range(0, len(self.data["DATE"])), self.data["DATE"], rotation=45)
        plt.show()
    
    def plotRatios(self):
        """Plot ratios"""
        fig = plt.figure(figsize=(30, 50))
        fig.subplots_adjust(hspace=0.8, wspace=0.25)

        for j in range(1, self.maxColumns - 1):
            x = self.data[self.data.columns[j]].copy()
            y = self.data[self.data.columns[j+1]].copy()

            # Slice
            x = x[self.start_offset:-self.end_offset]
            y = y[self.start_offset:-self.end_offset]
            
            # Define plot title, x-axis and y-axis labels
            pltXLabel = self.data.columns[j]
            pltYLabel = self.data.columns[j+1]
            pltTitle  = pltYLabel + "/" + pltXLabel

            # Plot data
            plt.subplot(4, 3, j)
            plt.plot(x, y, 'o', label=pltTitle)
            plt.title(pltTitle)
            plt.xlabel(pltXLabel)
            plt.ylabel(pltYLabel)

            # Built regression line
            # and plot it
            regY = []
            for regX in x:
                ratio_label = self.data.columns[j+1][:-3] + "/" + self.data.columns[j]
                r = self.result.loc[ratio_label, "ratio"]
                regY.append(r * regX )
            plt.plot(x, regY, color="blue")

        plt.show()                         
            
    def __dataProcessing(self):
        """pass"""
        # For each MIS
        for j in range(2, self.maxColumns):
            # Get intermediate MIS list
            imdMis = [k for k in range(1, j)]
            # Get intermediate ratios list
            imdRatios = [(k, k-1) for k in range(max(imdMis)+1, 1, -1)]
            # Calculate intermediate ratios
            self.__calculateRatio(imdRatios)
        
    def __calculateRatio(self, ratios):
        """Calculate ratio
           from ratio tuples
           ratios = [(1,2), (3,2)]"""
        # Calculate each ratio
        for ratio in ratios:
            # Define result as a dict
            result = dict()
            # if ratio already calculated then return
            ratio_label = self.__convertToMis(ratio)
            if ratio_label not in self.result.transpose():
                # Set regression data
                x = self.data.iloc[:, ratio[1]]
                y = self.data.iloc[:, ratio[0]]
                # Get result
                result[ratio_label] = [ratio] + self.__getRegResult(
                    x[self.start_offset:-self.end_offset],
                    y[self.start_offset:-self.end_offset])
                # Append new result to existing results
                newResult = pd.DataFrame.from_dict(result, orient='index',
                                                   columns=self.columnResult)
                self.result = pd.concat([self.result, newResult])
        self.result.sort_values(by=["ratio_tuple"], inplace=True)

    def __getRegResult(self, x, y):
        """Return results from
           model regression"""
        # If no data to regress, return NaN
        if x.isnull().all() or y.isnull().all():
            return([math.nan for i in range(1, len(self.columnResult))])
        # Set data for model
        modelData = pd.DataFrame({'x': x, 'y': y})
        # Set regression: y ~ x -1: "-1" to supress y-intercept term
        model = ols("y ~ x -1", modelData)
        # Fit model
        result = model.fit()
        # Get results and return
        coeff = result.params
        r_sq = result.rsquared
        std_err = result.bse
        conf = result.conf_int(0.05)
        pval = result.pvalues
        return([coeff['x'], r_sq, std_err['x'], pval['x'], conf[0]['x'],
                conf[1]['x'], min(x.count(), y.count())])

    def __forecastData(self):
        """Extrapolate MIS
           from real ratios"""
        self.forecast = self.data.copy()
        self.confinf = self.data.copy()
        self.confsup = self.data.copy()
        for j in range(2, self.maxColumns):
            last_index = self.forecast[self.data.columns[j]].last_valid_index()
            if last_index is None: last_index = 0
            for i in range(1+last_index, self.maxRows):
                ratio_label = self.forecast.columns[j][:-3] + "/" + self.forecast.columns[j-1]
                r = self.result.loc[ratio_label, "ratio"]
                rinf = self.result.loc[ratio_label, "conf_inf"]
                rsup = self.result.loc[ratio_label, "conf_sup"]
                v = self.forecast[self.forecast.columns[j-1]].values[i]
                self.forecast.loc[i, self.data.columns[j]] = float(r) * float(v)
                self.confinf.loc[i, self.data.columns[j]] = float(rinf) * float(v)
                self.confsup.loc[i, self.data.columns[j]] = float(rsup) * float(v)

    def __convertToMis(self, t):
        """ Convert ratio in tuple format to string """
        return self.data.columns[t[0]][:-3] + "/" + self.data.columns[t[1]]

    def __convertToMisTuple(self, ratio):
        """ Convert a ratio in string format to tupple """
        return tuple([self.data.columns.get_loc(m + "MIS")
                      for m in self.__splitRatio(ratio)])

    def __splitRatio(self, ratio):
        """Split ratio into mis
           and return the list"""
        return(ratio[:-3].split("/"))

    def __diffRatio(self, m):
        """Return the difference
           in month between two MIS"""
        return(int(m[0][:-3]) - int(m[1][:-3]))


def main():
    csvFilename = r"C:\Users\a033698\OneDrive - RENAULT\_GAMME_C\_SYNTHESE\JUPYTHER\XFA-EUROVEH.csv"

    r = ratioCore()
    r.getData(csvFilename)
    r.calculateRatio(["21/6MIS", "21/3MIS"])
    r.displayData()
    r.displayForecast()
    r.displayResult()
    r.plotMis(["1MIS", "3MIS", "6MIS", "9MIS", "12MIS"])
    r.plotConfMis(["12MIS", "15MIS", "18MIS", "21MIS", "21MIS", "24MIS"])
    r.plotRatios()

if __name__ == "__main__":
    main()
