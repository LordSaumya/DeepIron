pub mod data_loader;
pub mod linear_regression;
pub mod model;

#[cfg(test)]
mod tests {
    use super::*;
    use data_loader::*;
    use linear_regression::*;
    use model::model::Modeller;
    use polars::prelude::*;
    use model::loss_functions::{LossFunction, LossFunctionType};
    use std::path::Path;
    #[test]
    fn test_load_csv() {
        // Load the CSV file
        let path: &Path = Path::new("test/loadTest.csv");
        let result: Result<DataFrame, PolarsError> = data_loader::data_loader_util::load_csv(path);

        println!("{:?}", result);

        // Check if the result is Ok and the DataFrame is not empty
        assert!(result.is_ok());
        let df: DataFrame = result.expect("Failed to load CSV file");
        assert!(!df.is_empty());
    }

    #[test]
    fn test_transform_by_col_custom_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df.transform_cols(&["col1"], |s: &Series| s * 2).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[2.0, 4.0, 6.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_identity_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::identity())
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_power_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::power(2.0))
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 4.0, 9.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_log_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::log(2.0))
            .unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame =
            DataFrame::new(vec![Series::new("col1", &[0.0, 1.0, 2.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_split() {
        // Create a simple DataFrame for testing
        let df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Split the DataFrame
        let (train, test) = df.split(0.8).unwrap();

        // Check if the split is done correctly
        assert_eq!(train.height(), 4);
        assert_eq!(test.height(), 1);
    }

    #[test]
    fn test_z_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Z-normalise the column
        df = df.z_norm_cols(&["col1"]).unwrap();

        let std: f64 = f64::sqrt(2.0);
        let mean: f64 = 3.0;

        // Check if the z-normalisation is done correctly
        let expected_result = DataFrame::new(vec![Series::new(
            "col1",
            &[
                (1.0 - mean) / std,
                (2.0 - mean) / std,
                (3.0 - mean) / std,
                (4.0 - mean) / std,
                (5.0 - mean) / std,
            ],
        )])
        .unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_chained_transformations() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df
            .transform_cols(&["col1"], transformer_functions::power(2.0))
            .transform_cols(&["col1"], transformer_functions::log(2.0))
            .transform_cols(&["col1"], transformer_functions::identity())
            .z_norm_cols(&["col1"])
            .unwrap();

        let std: f64 = f64::sqrt(8.0 / 3.0);
        let mean: f64 = 2.0;

        // Check if the transformation is applied correctly
        let expected_result = DataFrame::new(vec![Series::new(
            "col1",
            &[(0.0 - mean) / std, (2.0 - mean) / std, (4.0 - mean) / std],
        )])
        .unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_min_max_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Min-max normalise the column
        df = df.min_max_norm_cols(&["col1"]).unwrap();

        // Check if the min-max normalisation is done correctly
        let expected_result =
            DataFrame::new(vec![Series::new("col1", &[0.0, 0.25, 0.5, 0.75, 1.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_mean_squared_error_loss_zero() {
        // Create a simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);

        // Compute the mean squared error
        let mse = LossFunctionType::MeanSquaredError.loss(&x, &x);

        // Check if the mean squared error is computed correctly
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_mean_squared_error_loss_non_zero() {
        // Create two simple series for testing
        let x: Series = Series::new("x", &[1.0, 2.0, 3.0]);
        let y: Series = Series::new("y", &[4.0, 5.0, 6.0]);

        // Compute the mean squared error
        let mse = LossFunctionType::MeanSquaredError.loss(&x, &y);

        // Check if the mean squared error is computed correctly
        assert_eq!(mse, 9.0);
    }

    #[test]
    fn test_mean_squared_error_gradient_zeros() {
        // Create a simple series and dataframe for testing
        let x: DataFrame = DataFrame::new(vec![Series::new("x1", &[1.0, 2.0, 3.0]), Series::new("x2", &[1.0, 2.0, 3.0])]).unwrap();
        let y: Series = Series::new("y", &[1.0, 2.0, 3.0]);


        // Compute the mean squared error
        let gradient = LossFunctionType::MeanSquaredError.gradient(&x, &y, &y);

        // Check if the mean squared error is computed correctly
        assert_eq!(gradient, Series::new("gradients", &[0.0, 0.0]));
    }

    #[test]
    fn test_mean_squared_error_gradient_non_zeros() {
        // Create a dataframe and two simple series for testing
        let x = DataFrame::new(vec![Series::new("x1", &[1.0, 2.0, 3.0]), Series::new("x2", &[4.0, 5.0, 6.0])]).unwrap();
        let y: Series = Series::new("y", &[1.0, 2.0, 3.0]);
        let y_pred: Series = Series::new("y_pred", &[4.0, 5.0, 6.0]);

        // Compute the mean squared error
        let gradient = LossFunctionType::MeanSquaredError.gradient(&x, &y, &y_pred);

        // Check if the mean squared error is computed correctly
        assert_eq!(gradient, Series::new("gradients", &[12.0, 30.0]));
    }

    #[test]
    fn test_linear_model_fit_predict() {
        // Sample data
        let x = DataFrame::new(vec![
            Series::new("feature1", vec![1, 2, 3]),
        ]);

        let y = Series::new("target", vec![10.0, 20.0, 30.0]);

        // Create a linear model
        let mut model = Linear::new(x.as_ref().unwrap().clone(), y.clone());

        // Fit the model
        assert!(model.fit(1000, 0.1).is_ok());

        // Predict using the same data
        let predictions = model.predict(&x.unwrap()).unwrap();

        // Print out the values for debugging
        println!("Predictions: {:?}", predictions);
        println!("Actual values: {:?}", y);

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        // Print out the sums for debugging
        println!("Sum of predictions: {:?}", predictions.sum::<f64>());
        println!("Sum of actual values: {:?}", y.sum::<f64>());

        // Check the sums
        assert!(
            (predictions.sum::<f64>().unwrap() - y.sum::<f64>().unwrap()).abs() < 1e-6,
            "Sums do not match within epsilon"
        );

        // Ensure predictions have the correct length
        assert_eq!(predictions.len(), y.len());

        assert_eq!(predictions.sum::<f64>(), y.sum()); // A simple example
    }
}
