use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Add;
use std::path::PathBuf;
use std::str::FromStr;
use tch::vision::dataset::Dataset;
use tch::Tensor;

#[derive(Debug, Clone, PartialEq)]
struct Data {
    row_number: u32,
    customer_id: String,
    surname: String,
    credit_score: f64,
    geography: String,
    gender: u8,
    age: f64,
    tenure: f64,
    balance: f64,
    num_of_products: f64,
    has_cr_card: f64,
    is_active_member: f64,
    estimated_salary: f64,
    exited: i8,
}

impl From<&String> for Data {
    fn from(value: &String) -> Self {
        let v: Vec<_> = value.split(',').map(|s| s).collect();

        Self {
            row_number: u32::from_str(v.get(0).unwrap()).unwrap(),
            customer_id: (v.get(1).unwrap()).to_string(),
            surname: (v.get(2).unwrap()).to_string(),
            credit_score: f64::from_str(v.get(3).unwrap()).unwrap(),
            geography: (v.get(4).unwrap()).to_string(),
            gender: if *v.get(5).unwrap() == "Female" { 0 } else { 1 },
            age: if let Some(x) = v.get(6) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            tenure: f64::from_str(v.get(7).unwrap()).unwrap(),
            balance: f64::from_str(v.get(8).unwrap()).unwrap(),
            num_of_products: f64::from_str(v.get(9).unwrap()).unwrap(),
            has_cr_card: if let Some(x) = v.get(10) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            is_active_member: if let Some(x) = v.get(11) {
                if x.len() > 0 {
                    f64::from_str(x).unwrap()
                } else {
                    f64::INFINITY
                }
            } else {
                0.
            },
            estimated_salary: f64::from_str(v.get(12).unwrap()).unwrap(),
            exited: if let Some(x) = v.get(13) {
                i8::from_str(*x).unwrap()
            } else {
                -1
            },
        }
    }
}

/// Gets train_data and test_data.
pub fn get_dataset(train_data_file: PathBuf, test_data_file: PathBuf) -> Dataset {
    let train_raw_data = fit_data(read_csv(train_data_file));

    // let mut geography_map = HashMap::new();

    let mut train_data_slice = vec![];
    train_raw_data.iter().for_each(|d| {
        train_data_slice.append(&mut vec![d.credit_score, d.gender as f64, d.age, d.tenure, d.balance, d.num_of_products, d.has_cr_card, d.is_active_member, d.estimated_salary])
    });

    let mut train_data = Tensor::from_slice(train_data_slice.as_slice());
    train_data = train_data.reshape([(train_data_slice.len() / 9) as i64, 9]);

    println!("{}", train_data);

    train_raw_data.into_iter().for_each(|d| {

    });

    unimplemented!();
}

/// Fills the holes in raw data.
///
/// Step1. Calculate average age.
///
/// Step2. Fill age holes with avg.
///
/// Step3. Fill has_cr_card holes with '0.0' and '1.0'.
///
/// Step4. Fill is_active_member holes with '0.0' and '1.0'.

fn fit_data(mut datas: Vec<Data>) -> Vec<Data> {
    let mut fitted_data = vec![];
    let mut valid_age_number = 0;
    let mut avg_age = 0.;

    datas.iter().for_each(|d| {
        if d.age != f64::INFINITY {
            valid_age_number += 1;
            avg_age += d.age;
        }
    });

    assert_ne!(valid_age_number, 0);
    avg_age /= valid_age_number as f64;

    datas.into_iter().for_each(|mut d| {
        if d.age == f64::INFINITY {
            d.age = avg_age;
        }

        fitted_data.append(&mut if d.has_cr_card == f64::INFINITY {
            d.has_cr_card = 0.0;
            if d.is_active_member == f64::INFINITY {
                d.is_active_member = 0.0;
                let mut u = d.clone();
                u.is_active_member = 1.0;

                let mut v = d.clone();
                v.has_cr_card = 1.0;
                let mut w = v.clone();
                w.is_active_member = 1.0;
                vec![d, u, v, w]
            } else {
                let mut u = d.clone();
                u.has_cr_card = 1.0;
                vec![d, u]
            }
        } else if d.is_active_member == f64::INFINITY {
            d.is_active_member = 0.0;
            let mut u = d.clone();
            u.is_active_member = 1.0;
            vec![d, u]
        } else {
            vec![d]
        });
    });

    fitted_data
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
            get_dataset(
                PathBuf::from("data/train.csv"),
                PathBuf::from("data/test.csv")
            );
    }

    #[test]
    fn test_read_csv() {
        read_csv(PathBuf::from("data/train.csv"));
        read_csv(PathBuf::from("data/test.csv"));
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
        let raw_data = vec![
            Data {
                row_number: 9001,
                customer_id: "15723217".to_string(),
                surname: "Cremonesi".to_string(),
                credit_score: 616.0,
                geography: "France".to_string(),
                gender: 1,
                age: 37.0,
                tenure: 9.0,
                balance: 0.0,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 0.0,
                estimated_salary: 111312.96,
                exited: -1,
            },
            Data {
                row_number: 9002,
                customer_id: "15733111".to_string(),
                surname: "Yeh".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 32.0,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 1.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
            Data {
                row_number: 9003,
                customer_id: "15733119".to_string(),
                surname: "Yee".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: f64::INFINITY,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: f64::INFINITY,
                is_active_member: f64::INFINITY,
                estimated_salary: 138759.15,
                exited: -1,
            },
        ];

        let mut fitted_data = vec![
            Data {
                row_number: 9001,
                customer_id: "15723217".to_string(),
                surname: "Cremonesi".to_string(),
                credit_score: 616.0,
                geography: "France".to_string(),
                gender: 1,
                age: 37.0,
                tenure: 9.0,
                balance: 0.0,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 0.0,
                estimated_salary: 111312.96,
                exited: -1,
            },
            Data {
                row_number: 9002,
                customer_id: "15733111".to_string(),
                surname: "Yeh".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 32.0,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 1.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
            Data {
                row_number: 9003,
                customer_id: "15733119".to_string(),
                surname: "Yee".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 34.5,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 0.0,
                is_active_member: 0.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
            Data {
                row_number: 9003,
                customer_id: "15733119".to_string(),
                surname: "Yee".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 34.5,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 0.0,
                is_active_member: 1.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
            Data {
                row_number: 9003,
                customer_id: "15733119".to_string(),
                surname: "Yee".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 34.5,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 0.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
            Data {
                row_number: 9003,
                customer_id: "15733119".to_string(),
                surname: "Yee".to_string(),
                credit_score: 688.0,
                geography: "Spain".to_string(),
                gender: 1,
                age: 34.5,
                tenure: 6.0,
                balance: 124179.3,
                num_of_products: 1.0,
                has_cr_card: 1.0,
                is_active_member: 1.0,
                estimated_salary: 138759.15,
                exited: -1,
            },
        ]
        .into_iter();

        fit_data(raw_data).into_iter().for_each(|d| {
            assert_eq!(fitted_data.next().unwrap(), d);
        });
        assert!(fitted_data.next().is_none());
    }
}
