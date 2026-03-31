use crate::tensor::TensorShape;

#[test]
fn shape_computes_row_major_strides() {
    let shape = TensorShape::new(vec![3, 4, 5]).unwrap();
    assert_eq!(shape.strides(), &[20, 5, 1]);
    assert_eq!(shape.elem_count(), 60);
}

#[test]
fn flat_index_round_trip_is_stable() {
    let shape = TensorShape::new(vec![3, 4, 5]).unwrap();
    let flat = shape.index_to_flat(&[2, 1, 4]).unwrap();
    let restored = shape.flat_to_index(flat);

    assert_eq!(flat, 49);
    assert_eq!(restored, vec![2, 1, 4]);
}

#[test]
fn transpose_swaps_rows_and_columns() {
    let shape = TensorShape::from_2d(7, 11).unwrap();
    let transposed = shape.transpose_2d().unwrap();

    assert_eq!(transposed.dims(), &[11, 7]);
}
