use crate::data_loader::DataFrameTransformer;
use crate::model::loss_functions::{LossFunction, LossFunctionType};
use crate::model::activation_functions::{ActivationFunction, ActivationFunctionType};
use crate::model::*;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a logistic model.
///
/// # Example
///
/// ```
/// let model = Model::Logistic::new();
///
/// model.fit(&x, &y);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub struct Logistic {
    // Fields for training
    pub x: DataFrame,
    pub y: Series,
    pub loss_function: LossFunctionType,
    pub activation_function: ActivationFunctionType,

    // Fields to store results
    pub intercept: f64,
    pub coefficients: Vec<f64>,
}

impl Logistic {
    /// Create a new Logistic model.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::Logistic::new();
    /// ```
    pub fn new(x: DataFrame, y: Series) -> Logistic {
        let x_width = x.width();
        Logistic {
            x: x,
            y: y,
            loss_function: LossFunctionType::BinaryCrossEntropy,
            activation_function: ActivationFunctionType::Sigmoid,
            intercept: 0.0,
            coefficients: vec![0.0; x_width],
        }
    }

    pub fn round_predictions(predictions: &Series) -> Series {
        let mut rounded_predictions: Vec<f64> = Vec::with_capacity(predictions.len());

        for prediction in predictions.f64().unwrap().into_iter() {
            let prediction: f64 = prediction.unwrap();
            if prediction > 0.5 {
                rounded_predictions.push(1.0);
            } else {
                rounded_predictions.push(0.0);
            }
        }

        Series::new("prediction", rounded_predictions)
    }

    fn compute_gradients(&self, predictions: &Series) -> (f64, Vec<f64>) {
        let error: Series = &self.y - predictions;
        let mut gradients: Vec<f64> = Vec::with_capacity(self.coefficients.len());
        let intercept_gradient: f64 = error.mean().unwrap() * -2.0;

        for (_i, _) in self.coefficients.iter().enumerate() {
            let gradient: f64 = self
                .loss_function
                .gradient(&self.x, &self.y, predictions)
                .mean()
                .unwrap();
            gradients.push(gradient);
        }

        (intercept_gradient, gradients)
    }
}

impl model::Modeller for Logistic {
    fn fit(&mut self, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError> {
        // Check if data are valid
        if self.x.shape().0 != self.y.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shape mismatch between X and y".into(),
            ));
        }

        for _ in 0..num_epochs {
            let predictions: Series = self.predict(&self.x)?;
            let gradients: (f64, Vec<f64>) = self.compute_gradients(&predictions);

            self.intercept -= learning_rate * gradients.0;

            for (i, coef) in self.coefficients.iter_mut().enumerate() {
                *coef -= learning_rate * gradients.1[i];
            }
        }

        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError> {
        let mut predictions: Series = Series::new("prediction", vec![self.intercept; x.height()]);

        for (i, coef) in self.coefficients.iter().enumerate() {
            let feature_values: &Series = &x.get_col_by_index(i).unwrap();
            predictions = feature_values * *coef + predictions;
        }

        // Apply and return the provided activation function
        Ok(self.activation_function.activate(&predictions))
    }

    fn accuracy(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError> {
        let y_pred: Series = self.predict(x)?;
        let y_pred_rounded: Series = Logistic::round_predictions(&y_pred);
        let correct_predictions: Series = y_pred_rounded.equal(y).unwrap().into_series();
        let num_correct_predictions: f64 = correct_predictions.sum().unwrap();
        let accuracy: f64 = num_correct_predictions / y_pred_rounded.len() as f64;
        Ok(accuracy)
    }

    fn loss(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError> {
        let y_pred = self.predict(x)?;
        Ok(self.loss_function.loss(y, &y_pred))
    }
}