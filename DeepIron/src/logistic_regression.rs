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
/// model.fit(num_epochs, learning_rate);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub struct Logistic {
    // Fields for training
    pub loss_function: LossFunctionType,

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
    pub fn new() -> Logistic {
        Logistic {
            loss_function: LossFunctionType::BinaryCrossEntropy,
            intercept: 0.0,
            coefficients: Vec::new(),
        }
    }

    pub fn classify(predictions: &Series, threshold: f64) -> Series {
        let mut classified_preds: Vec<f64> = Vec::with_capacity(predictions.len());

        for prediction in predictions.f64().unwrap().into_iter() {
            let prediction: f64 = prediction.unwrap();
            if prediction > threshold {
                classified_preds.push(1.0);
            } else {
                classified_preds.push(0.0);
            }
        }

        Series::new("classified predictions", classified_preds)
    }

    fn compute_gradients(&self, x: &DataFrame, y: &Series, predictions: &Series) -> (f64, Vec<f64>) {
        let mut gradients: Vec<f64> = Vec::with_capacity(self.coefficients.len());
        let intercept_gradient: f64 = self.loss_function.intercept_gradient(y, predictions);

        for (_i, _) in self.coefficients.iter().enumerate() {
            let gradient: f64 = self
                .loss_function
                .gradient(x, y, predictions)
                .mean()
                .unwrap();
            gradients.push(gradient);
        }

        (intercept_gradient, gradients)
    }
}

impl model::Modeller for Logistic {
    fn fit(&mut self, x: &DataFrame, y: &Series, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError> {
        // Check if data are valid
        if x.shape().0 != y.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shape mismatch between X and y".into(),
            ));
        }

        for _ in 0..num_epochs {
            let predictions: Series = self.predict(x)?;
            let gradients: (f64, Vec<f64>) = self.compute_gradients(x, y, &predictions);

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

        Ok(ActivationFunctionType::Sigmoid.activate(&predictions))
    }

    fn accuracy(&self, x: &DataFrame, y: &Series) -> Result<f64, PolarsError> {
        let y_pred: Series = self.predict(x)?;
        let y_pred_rounded: Series = Logistic::classify(&y_pred, 0.5);
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