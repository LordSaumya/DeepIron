use std::ops::Add;

use crate::dataLoader::DataFrameTransformer;
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
            x: x,
            y: y,
            lossFunction: LossFunctionType::MeanSquaredError,
            intercept: 0.0,
            coefficients: vec![],
        }
    }

    fn computeGradients(&self, predictions: &Series) -> (f64, Vec<f64>) {
        // Use given loss function to compute gradients
        let mut gradients: Vec<f64> = Vec::with_capacity(self.coefficients.len());
        let mut intercept_gradient: f64 = 0.0;
    
        for (i, coef) in self.coefficients.iter().enumerate() {
            let feature_values: &Series = &self.x.getColByIndex(i).unwrap();
            let gradient: f64 = self.lossFunction.gradient(&feature_values, predictions).mean().unwrap();
            gradients.push(gradient);
        }
    
        // Compute the intercept gradient
        intercept_gradient = self.lossFunction.gradient(&self.y, predictions).mean().unwrap();
    
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
            let feature_values: &Series = &x.getColByIndex(i).unwrap();
            predictions = predictions + (feature_values * *coef);
        }
        
        Ok(predictions)
    }
}
