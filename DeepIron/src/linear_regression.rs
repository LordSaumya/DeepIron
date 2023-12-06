use crate::data_loader::DataFrameTransformer;
use crate::model::*;
use crate::model::LossFunctions::{LossFunction, LossFunctionType};
use polars::error::ErrString;
use polars::error::PolarsError::ShapeMismatch;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a linear model.
///
/// # Example
///
/// ```
/// let model = Model::Linear::new();
///
/// model.fit(&x, &y);
///
/// let y_pred = model.predict(&x);
///
/// ```
pub struct Linear {
    // Fields for training
    pub x: DataFrame,
    pub y: Series,
    pub lossFunction: LossFunctionType,

    // Fields to store results
    pub intercept: f64,
    pub coefficients: Vec<f64>,
}

impl Linear {
    /// Create a new Linear model.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::Linear::new();
    /// ```
    pub fn new(x: DataFrame, y: Series) -> Linear {
        Linear {
            x: x.clone(),
            y: y,
            lossFunction: LossFunctionType::MeanSquaredError,
            intercept: 0.0,
            coefficients: vec![0.0; x.width()],
        }
    }

    fn computeGradients(&self, predictions: &Series) -> (f64, Vec<f64>) {
        let error: Series = &self.y - predictions;
        let mut gradients: Vec<f64> = Vec::with_capacity(self.coefficients.len());
        let intercept_gradient: f64 = error.mean().unwrap() * -2.0;

        for (i, _) in self.coefficients.iter().enumerate() {
            let gradient: f64 = self.lossFunction.gradient(&self.x, &self.y, predictions).mean().unwrap();
            gradients.push(gradient);
        }

        (intercept_gradient, gradients)
    }
}

impl Model::Modeller for Linear {
    fn fit(&mut self, num_epochs: u32, learning_rate: f64) -> Result<(), PolarsError> {
        // Check if data are valid
        if self.x.shape().0 != self.y.len() {
            return Err(PolarsError::ShapeMismatch("Shape mismatch between X and y".into()));
        }
        
        for _ in 0..num_epochs {
            let predictions = self.predict(&self.x)?;
            let gradients = self.computeGradients(&predictions);

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
        
        Ok(predictions)
    }
}
