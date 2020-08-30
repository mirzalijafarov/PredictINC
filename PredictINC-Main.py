# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
import random



class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("PredictINC-GUI.ui", self)

        self.setWindowTitle("PredictINC")

        #self.pushButton_clear.clicked.connect(self.delete_graph)
        #self.actionLoad.triggered.connect(self.update_graph)
        self.pushButton_load_and_plot.clicked.connect(self.load_and_plot)
        self.pushButton_linear.clicked.connect(self.linear_regression)
        self.pushButton_poly.clicked.connect(self.polynomial_regression)
        self.pushButton_clear.clicked.connect(self.delete_graph)
        self.pushButton_predict.clicked.connect(self.predict)

        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def load_and_plot(self):
        global x
        global y
        options1 = QFileDialog.Options()
        filename1, filter1 = QFileDialog.getOpenFileName(None, caption='Open file',
                                                         filter='CSV files (*.csv)', options=options1)
        if filename1:
            survey_path = filename1
            survey = pd.read_csv(survey_path, skiprows=[0, 2])
            # survey = survey.drop(["Unnamed: 26"], axis = 1)

            # Collar Coordinates
            collar_easting = survey["Easting"].iloc[0]
            collar_northing = survey["Northing"].iloc[0]
            collar_elevation = survey["Elevation"].iloc[0]

            # Collar Parameters
            collar_dip = float(self.lineEdit_dip.text())
            collar_azimuth = float(self.lineEdit_azimuth.text())

            # Calculating Target Coordinates
            target_easting = ((survey["Station"]
                              ) / 2 * ((np.sin(np.radians(collar_dip + 90)) * np.sin(np.radians(collar_azimuth))
                                        ) + (np.sin(np.radians(collar_dip + 90)) * np.sin(np.radians(collar_azimuth))))
                              ) + collar_easting
            target_northing = ((survey["Station"]
                               ) / 2 * ((np.sin(np.radians(collar_dip + 90)) * np.cos(np.radians(collar_azimuth))
                                         ) + (np.sin(np.radians(collar_dip + 90)) * np.cos(np.radians(collar_azimuth))))
                               ) + collar_northing

            target_elevation = (-1 * (survey["Station"]
            ) / 2 * (np.cos(np.radians(collar_dip + 90)) + np.cos(np.radians(collar_dip + 90)))
                                ) + collar_elevation

            # Actual Coordinates
            actual_easting = survey["Easting"]
            actual_northing = survey["Northing"]
            actual_elevation = survey["Elevation"]

            # Target Differences
            targ_east_differ = abs(target_easting - actual_easting)
            targ_nort_differ = abs(target_northing - actual_northing)
            targ_elev_differ = abs(target_elevation - actual_elevation)

            tot_misc_to_targ = np.sqrt(targ_east_differ ** 2 + targ_nort_differ ** 2 + targ_elev_differ ** 2)
            survey["Deviation"] = tot_misc_to_targ

            x = survey.Station.values.reshape((-1, 1))
            y = survey["Deviation"].values

            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.scatter(x, y, s=2)
            #self.MplWidget.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
            self.MplWidget.canvas.axes.set_title('Depth - Deviation')
            self.MplWidget.canvas.draw()



    def linear_regression(self):
        regressor = LinearRegression()
        regressor.fit(x, y)
        r_sq = regressor.score(x, y)
        self.label_r2.setText("R2: " + str(round(r_sq, 2)))

        #self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(x, regressor.predict(x))
        # self.MplWidget.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
        #self.MplWidget.canvas.axes.set_title('Depth - Deviation')
        self.MplWidget.canvas.draw()


    def linear_regression_predict(self):
        regressor = LinearRegression()
        regressor.fit(x, y)
        depth_input = float(self.lineEdit_depth.text())
        depth = np.array([depth_input]).reshape(-1, 1)
        linear_y_pred = regressor.predict(depth)
        self.label_deviation.setText('Deviation: ' + str(round(linear_y_pred[0], 2)) + ' m')




    def polynomial_regression(self):
        transformer = PolynomialFeatures(degree=2, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        poly_model = LinearRegression().fit(x_, y)
        poly_r_sq = poly_model.score(x_, y)
        self.label_r2.setText("R2: " + str(round(poly_r_sq, 2)))

        self.MplWidget.canvas.axes.plot(x_[:, 0], poly_model.predict(x_))
        self.MplWidget.canvas.draw()


    def polynomial_regression_predict(self):
        transformer = PolynomialFeatures(degree=2, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        poly_model = LinearRegression().fit(x_, y)
        depth_input = float(self.lineEdit_depth.text())
        depth = np.array([depth_input, depth_input**2]).reshape(1, 2)
        poly_y_pred = poly_model.predict(depth)
        self.label_deviation.setText('Deviation: ' + str(round(poly_y_pred[0], 2)) + ' m')








    def predict(self):
        if self.checkBox_linear.isChecked():
            self.linear_regression_predict()
        elif self.checkBox_poly.isChecked():
            self.polynomial_regression_predict()
        else:
            self.label_deviation.setText('Please, select a prediction method')








    def delete_graph(self):
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.draw()


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()

