//! A set of structs and functions for k-means clustering.

use crate::data_loader::DataFrameTransformer;
use crate::model::activation_functions::{ActivationFunction, ActivationFunctionType};
use crate::model::loss_functions::{LossFunction, LossFunctionType};
use crate::model::*;
use polars::prelude::*;
use polars::series::Series;

/// A struct that defines a k-means clustering model
///
/// # Example
///
/// ```
/// 
/// let model = Model::KMeans::new_random(3, EndCondition::MaxIter(100));
/// 
/// model.fit(&x);
/// 
/// let  = model.predict(&x);
///
/// ```
pub struct KMeans {
    pub init_type: InitType,
    pub n_clusters: usize,
    pub centroid_coordinates: DataFrame,
    pub end_condition: EndCondition,
}

#[derive(Clone)]
/// An enum that defines the types of initialisation for the positions of the centroids in the k-means model.
pub enum InitType {
    /// Randomly initialise the centroids.
    Random,
    /// Use user-defined initial centroids.
    UserDefined(DataFrame),
    /// Initialise the centroids to be equidistant from each other.
    Equidistant,
}

#[derive(Clone)]
/// An enum that defines the conditions for ending the k-means algorithm.
pub enum EndCondition {
    /// The maximum number of iterations.
    /// The algorithm will stop when the centroids have converged or the maximum number of iterations is reached.
    MaxIter(usize),
    /// The tolerance for the change in the centroids.
    /// The algorithm will stop when the change in the centroids is less than the tolerance.
    Tol(f64),
}

impl KMeans {
    /// Create a new k-means model with the specified number of randomly initialised clusters and specified end condition.
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - The number of clusters to create.
    /// 
    /// * `end_condition` - The condition for ending the k-means algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// let model = Model::KMeans::new(3, EndCondition::MaxIter(100));
    /// ```
    pub fn new_random(n_clusters: usize, end_condition: EndCondition) -> KMeans {
        KMeans {
            init_type: InitType::Random,
            n_clusters,
            centroid_coordinates: DataFrame::new(vec![]).unwrap(),
            end_condition
        }
    }

    /// Create a new k-means model with the specified number of clusters, user-defined initial centroids, and specified end condition.
    /// 
    /// # Arguments
    /// 
    /// * `n_clusters` - The number of clusters to create.
    /// 
    /// * `centroids` - A DataFrame containing the initial centroid coordinates.
    /// 
    /// * `end_condition` - The condition for ending the k-means algorithm.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = Model::KMeans::new_user_defined(3, centroids, EndCondition::MaxIter(100));
    /// 
    /// ```
    pub fn new_user_defined(n_clusters: usize, centroids: DataFrame, end_condition: EndCondition) -> KMeans {
        KMeans {
            init_type: InitType::UserDefined(centroids),
            n_clusters,
            centroid_coordinates: DataFrame::new(vec![]).unwrap(),
            end_condition
        }
    }

    /// Create a new k-means model with the specified number of clusters, equidistant initial centroids, and specified end condition.
    /// 
    /// # Arguments
    /// 
    /// * `n_clusters` - The number of clusters to create.
    /// 
    /// * `end_condition` - The condition for ending the k-means algorithm.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let model = Model::KMeans::new_equidistant(3, EndCondition::MaxIter(100));
    /// 
    /// ```
    pub fn new_equidistant(n_clusters: usize, end_condition: EndCondition) -> KMeans {
        KMeans {
            init_type: InitType::Equidistant,
            n_clusters,
            centroid_coordinates: DataFrame::new(vec![]).unwrap(),
            end_condition
        }
    }

    /// Calculate the Euclidean distance between two points.
    /// 
    /// # Arguments
    /// 
    /// * `point1` - A Series containing the coordinates of the first point.
    /// 
    /// * `point2` - A Series containing the coordinates of the second point.
    /// 
    /// # Returns
    /// 
    /// * `f64` - The Euclidean distance between the two points.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let distance = Model::KMeans::euclidean_distance(&point1, &point2);
    /// 
    /// ```
    fn euclidean_distance(point1: &Series, point2: &Series) -> f64 {
        let point1: Vec<f64> = point1.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();
        let point2: Vec<f64> = point2.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();

        let mut sum: f64 = 0.0;

        for (i, _) in point1.iter().enumerate() {
            sum += (point1[i] - point2[i]).powi(2);
        }
        sum.sqrt()
    }

    /// Check for convergence of the centroids.
    /// 
    /// # Arguments
    /// 
    /// * `old_centroids` - A DataFrame containing the old centroid coordinates.
    /// 
    /// * `new_centroids` - A DataFrame containing the new centroid coordinates.
    /// 
    /// * `tol` - The tolerance for the change in the centroids.
    /// 
    /// # Returns
    /// 
    /// * `bool` - A boolean indicating whether the centroids have converged.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let converged = Model::KMeans::check_convergence(&old_centroids, &new_centroids, 0.01);
    /// 
    /// ```
    fn check_convergence(old_centroids: &DataFrame, new_centroids: &DataFrame, tol: f64) -> bool {
        let mut converged: bool = true;

        for (i, _) in old_centroids.columns().iter().enumerate() {
            let old_centroid: &Series = old_centroids.column(i).unwrap();
            let new_centroid: &Series = new_centroids.column(i).unwrap();

            let old_centroid: Vec<f64> = old_centroid.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();
            let new_centroid: Vec<f64> = new_centroid.f64().unwrap().into_iter().map(|x| x.unwrap()).collect();

            for (j, _) in old_centroid.iter().enumerate() {
                if (old_centroid[j] - new_centroid[j]).abs() > tol {
                    converged = false;
                    break;
                }
            }
        }
        converged
    }

    /// Initialise the centroids for a k-means model.
    /// 
    /// # Arguments
    /// 
    /// * `x` - A DataFrame of features.
    /// 
    /// # Returns
    /// 
    /// * `Result<(), PolarsError>` - A Result indicating whether the centroids were successfully initialised.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// model = Model::KMeans::new_random(3);
    /// model.initialise_centroids(&x);
    /// 
    /// ```
    fn initialise_centroids(&mut self, x: &DataFrame) -> Result<(), PolarsError> {
        let mut centroids: DataFrame = DataFrame::new(vec![]).unwrap();

        /// Find min and max for each column (feature)
        let mut min: Vec<f64> = Vec::with_capacity(x.width());
        let mut max: Vec<f64> = Vec::with_capacity(x.width());
        for col in x.columns() {
            min.push(col.min().unwrap().unwrap());
            max.push(col.max().unwrap().unwrap());
        }

        match self.init_type {
            InitType::Random => {
                for _ in 0..self.n_clusters {
                    let mut centroid: Vec<f64> = Vec::with_capacity(x.width());
                    for _ in 0..x.width() {
                        centroid.push(rand::random::<f64>() * (max[0] - min[0]) + min[0]);
                    }
                    centroids.with_column(Series::new("centroid", centroid))?;
                }
                self.centroid_coordinates = centroids;
            }

            InitType::UserDefined(user_centroids) => {
                if user_centroids.height() != self.n_clusters {
                    panic!("Number of user-defined centroids does not match number of clusters");
                }
                if user_centroids.width() != x.width() {
                    panic!("Number of features in user-defined centroids does not match number of features in data");
                }
                self.centroid_coordinates = user_centroids;
            }

            InitType::Equidistant => {
                let equidist: Vec<f64> = (max - min).iter().map(|x| x / (n_clusters as f64)).collect();
                for i in 0..self.n_clusters {
                    let mut centroid: Vec<f64> = Vec::with_capacity(x.width());
                    for j in 0..x.width() {
                        centroid.push(min[j] + (i as f64) * equidist[j]);
                    }
                    centroids.with_column(Series::new("centroid", centroid))?;
                }
                self.centroid_coordinates = centroids;
            }
        }
    }

    /// Assigns a cluster to each data point based on the current centroid coordinates.
    /// 
    /// # Arguments
    /// 
    /// * `x` - A DataFrame of features.
    /// 
    /// # Returns
    /// 
    /// * `Vec<usize>` - A vector containing the cluster assignments for each data point.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let cluster_assignments = model.assign_clusters(&x);
    /// 
    /// ```
    fn assign_clusters(&self, x: &DataFrame) -> Vec<usize> {
        let mut cluster_assignments: Vec<usize> = Vec::with_capacity(x.height());
        for i in 0..x.height() {
            let mut min_distance: f64 = f64::INFINITY;
            let mut cluster: usize = 0;
            for j in 0..self.n_clusters {
                let distance: f64 = KMeans::euclidean_distance(&x.row(i), &self.centroid_coordinates.row(j));
                if distance < min_distance {
                    min_distance = distance;
                    cluster = j;
                }
            }
            cluster_assignments.push(cluster);
        }
        cluster_assignments
    }
}

impl model::ClusterModeller for KMeans {
    /// Fit the model to the data.
    /// 
    /// # Arguments
    /// 
    /// * `x` - A DataFrame of features.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// model.fit(&x);
    /// 
    /// ```
    /// 
    /// # Returns
    /// 
    /// * `Result<(), PolarsError>` - A Result indicating whether the model was successfully fitted.
    fn fit(
        &mut self,
        x: &DataFrame,
    ) -> Result<(), PolarsError> {

        self.initialise_centroids(x)?;

        let mut old_centroids: DataFrame = DataFrame::new(vec![]).unwrap();
        let mut new_centroids: DataFrame = DataFrame::new(vec![]).unwrap();

        match self.end_condition {
            EndCondition::MaxIter(max_iter) => {
                for _ in 0..max_iter {
                    old_centroids = self.centroid_coordinates.clone();
                    let cluster_assignments: Vec<usize> = self.assign_clusters(x);
                    for i in 0..self.n_clusters {
                        let cluster_indices: Vec<usize> = cluster_assignments.iter().enumerate().filter(|(_, &x)| x == i).map(|(x, _)| x).collect();
                        let cluster_data: DataFrame = x.select_rows(cluster_indices.as_slice())?;
                        let mut new_centroid: Vec<f64> = Vec::with_capacity(x.width());
                        for j in 0..x.width() {
                            new_centroid.push(cluster_data.column(j).unwrap().mean().unwrap().unwrap());
                        }
                        new_centroids.with_column(Series::new("centroid", new_centroid))?;
                    }
                    self.centroid_coordinates = new_centroids.clone();
                    if KMeans::check_convergence(&old_centroids, &new_centroids, 0.0) {
                        break;
                    }
                }
            }

            EndCondition::Tol(tol) => {
                loop {
                    old_centroids = self.centroid_coordinates.clone();
                    let cluster_assignments: Vec<usize> = self.assign_clusters(x);
                    for i in 0..self.n_clusters {
                        let cluster_indices: Vec<usize> = cluster_assignments.iter().enumerate().filter(|(_, &x)| x == i).map(|(x, _)| x).collect();
                        let cluster_data: DataFrame = x.select_rows(cluster_indices.as_slice())?;
                        let mut new_centroid: Vec<f64> = Vec::with_capacity(x.width());
                        for j in 0..x.width() {
                            new_centroid.push(cluster_data.column(j).unwrap().mean().unwrap().unwrap());
                        }
                        new_centroids.with_column(Series::new("centroid", new_centroid))?;
                    }
                    self.centroid_coordinates = new_centroids.clone();
                    if KMeans::check_convergence(&old_centroids, &new_centroids, tol) {
                        break;
                    }
                }
            }
        }
    }

    /// Predict the cluster assignments for a DataFrame of features.
    /// 
    /// # Arguments
    /// 
    /// * `x` - A DataFrame of features.
    /// 
    /// # Returns
    /// 
    /// * `Result<Series, PolarsError>` - A result containing the predicted target values.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let cluster_assignments = model.predict(&x);
    /// 
    /// ```
    fn predict(&self, x: &DataFrame) -> Result<Series, PolarsError> {
        let cluster_assignments: Vec<usize> = self.assign_clusters(x);
        let cluster_assignments: Vec<f64> = cluster_assignments.iter().map(|x| *x as f64).collect();
        Ok(Series::new("clusters", cluster_assignments))
    }

    /// Calculate the compactness of the clusters using sum of squared errors.
    /// 
    /// # Arguments
    /// 
    /// * `x` - A DataFrame of features.
    /// 
    /// # Returns
    /// 
    /// * `Result<f64, PolarsError>` - A result containing the compactness of the clusters.
    /// 
    /// # Example
    /// 
    /// ```
    /// 
    /// let compactness = model.compactness(&x);
    /// 
    /// ```
    fn compactness(&self, x: &DataFrame) -> Result<f64, PolarsError> {
        let cluster_assignments: Vec<usize> = self.assign_clusters(x);
        let mut compactness: f64 = 0.0;
        for i in 0..x.height() {
            let cluster: usize = cluster_assignments[i];
            let distance: f64 = KMeans::euclidean_distance(&x.row(i), &self.centroid_coordinates.row(cluster));
            compactness += distance.powi(2);
        }
        Ok(compactness)
    }
}

