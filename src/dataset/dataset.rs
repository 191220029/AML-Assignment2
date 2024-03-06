use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;
use tch::vision::dataset::Dataset;
use tch::Tensor;

#[derive(Debug, Eq)]
struct Data {
    RowNumber: u32,
    CustomerId: String,
    Surname: String,
    CreditScore: f64,
    Geography: String,
    Gender: u8,
    Age: f64,
    Tenure: f64,
    Balance: f64,
    NumOfProducts: f64,
    HasCrCard: f64,
    IsActiveMember: f64,
    EstimatedSalary: f64,
    Exited: i8,
}

impl From<&String> for Data {
    fn from(value: &String) -> Self {
        let v: Vec<_> = value.split(',').map(|s| s).collect();

        Self {
            RowNumber: u32::from_str(v.get(0).unwrap()).unwrap(),
            CustomerId: (v.get(1).unwrap()).to_string(),
            Surname: (v.get(2).unwrap()).to_string(),
            CreditScore: f64::from_str(v.get(3).unwrap()).unwrap(),
            Geography: (v.get(4).unwrap()).to_string(),
            Gender: if *v.get(5).unwrap() == "Female" { 0 } else { 1 },
            Age: if let Some(x) = v.get(6) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            Tenure: f64::from_str(v.get(7).unwrap()).unwrap(),
            Balance: f64::from_str(v.get(8).unwrap()).unwrap(),
            NumOfProducts: f64::from_str(v.get(9).unwrap()).unwrap(),
            HasCrCard: if let Some(x) = v.get(10) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            IsActiveMember: if let Some(x) = v.get(11) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            EstimatedSalary: f64::from_str(v.get(12).unwrap()).unwrap(),
            Exited: if let Some(x) = v.get(13) {
                i8::from_str(*x).unwrap()
            } else {
                -1
            },
        }
    }
}

/// Gets train_data and test_data.
pub fn get_dataset(train_data_file: PathBuf, test_data_file: PathBuf) -> Dataset {
    let train_data = read_csv(train_data_file);

    let mut train_data = Tensor::new();
    let mut train_label = Tensor::new();
    unimplemented!();
}

// Fills the holes in raw data.
fn fit_data(datas: &mut Vec<Data>) {
    unimplemented!()
}

fn read_csv(path: PathBuf) -> Vec<Data> {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let mut buf = String::new();

    // remove title
    reader.read_line(&mut buf).unwrap();
    buf.clear();

    let mut v = vec![];

    while let Ok(len) = reader.read_line(&mut buf) {
        if len > 0 {
            v.push(Data::from(&buf.trim().to_string()));
            buf.clear();
        } else {
            break;
        }
    }

    v
}

#[cfg(test)]
mod test_dataset {
    use crate::dataset::dataset::{fit_data, get_dataset, read_csv, Data};
    use std::path::PathBuf;

    #[test]
    fn test_get_dataset() {
        println!(
            "{:?}",
            get_dataset(
                PathBuf::from("data/train.csv"),
                PathBuf::from("data/test.csv")
            )
        )
    }

    #[test]
    fn test_read_csv() {
        read_csv(PathBuf::from("data/train.csv"));
    }

    #[test]
    fn test_from_str() {
        let d1 = String::from("9001,15723217,Cremonesi,616,France,Male,37,9,0,1,1,0,111312.96");
        let d2 = String::from("9002,15733111,Yeh,688,Spain,Male,32,6,124179.3,1,1,1,138759.15");
        println!("{:?}", Data::from(&d1));
        println!("{:?}", Data::from(&d2));
    }

    #[test]
    fn test_fit_data() {
        assert_eq!(vec![], fit_data(&mut vec![]));
    }
}
