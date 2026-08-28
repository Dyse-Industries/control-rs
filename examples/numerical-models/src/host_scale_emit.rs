//! Host-scale native JSON artifact emitters.

use std::path::PathBuf;
use std::time::Instant;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::state_space::ArrayStateSpace;
use control_rs::tensor::ArrayTensor;
use control_rs::transfer_function::ArrayTransferFunction;
use serde_json::{Value, json};

use crate::emit::{manifest_dir, write_artifact};
use crate::solve_residual_ratio;

const ITERS_1024: u32 = 1;
const BIG_STACK: usize = 256 * 1024 * 1024;

fn load_python(rel: &str) -> Value {
    let path = manifest_dir().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing host-scale python artifact: {}\nrun: cargo run --example all",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("parse")
}

fn fill_matrix_json<const R: usize, const C: usize>(v: &Value) -> Owned<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut m = Owned::<f64, R, C>::zero();
    let rows = v.as_array().expect("matrix rows");
    for i in 0..R {
        let row = rows[i].as_array().expect("row");
        for j in 0..C {
            m.set(i, j, row[j].as_f64().unwrap()).expect("set");
        }
    }
    m
}

fn fill_matrix_json_n<const N: usize>(v: &Value) -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    fill_matrix_json::<N, N>(v)
}

fn col_from_json<const N: usize>(v: &Value) -> Owned<f64, N, 1>
where
    Const<N>: Dim,
{
    let mut m = Owned::<f64, N, 1>::zero();
    let rows = v.as_array().expect("col rows");
    for i in 0..N {
        m.set(i, 0, rows[i][0].as_f64().unwrap()).expect("set");
    }
    m
}

fn col_to_json<const N: usize>(m: &Owned<f64, N, 1>) -> Value
where
    Const<N>: Dim,
{
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        rows.push(json!([m.get(i, 0).copied().unwrap_or(0.0)]));
    }
    Value::Array(rows)
}

fn rel_err_col<const N: usize>(got: &Owned<f64, N, 1>, gold: &Owned<f64, N, 1>) -> f64
where
    Const<N>: Dim,
{
    let mut diff = 0.0_f64;
    let mut goldn = 0.0_f64;
    for i in 0..N {
        let g = gold.get(i, 0).copied().unwrap();
        diff = diff.max((got.get(i, 0).copied().unwrap() - g).abs());
        goldn = goldn.max(g.abs());
    }
    if goldn == 0.0 { diff } else { diff / goldn }
}

fn rel_err_mat<const N: usize>(got: &Owned<f64, N, N>, gold: &Owned<f64, N, N>) -> f64
where
    Const<N>: Dim,
{
    let mut diff = 0.0_f64;
    let mut goldn = 0.0_f64;
    for i in 0..N {
        let mut ds = 0.0_f64;
        let mut gs = 0.0_f64;
        for j in 0..N {
            let g = gold.get(i, j).copied().unwrap();
            ds += (got.get(i, j).copied().unwrap() - g).abs();
            gs += g.abs();
        }
        if ds > diff {
            diff = ds;
        }
        if gs > goldn {
            goldn = gs;
        }
    }
    if goldn == 0.0 { diff } else { diff / goldn }
}

fn well_conditioned<const N: usize>(scale: f64) -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::from_fn(|i, j| {
        let diag = if i == j { 1.0 } else { 0.0 };
        diag + scale / ((i + j + 1) as f64)
    })
}

fn emit_hilbert12() {
    emit_hilbert_fixed::<12>("hilbert12");
}

fn emit_hilbert16() {
    emit_hilbert_fixed::<16>("hilbert16");
}

fn emit_hilbert128() {
    emit_hilbert_fixed::<128>("hilbert128");
}

fn emit_hilbert_fixed<const N: usize>(name: &str)
where
    Const<N>: Dim,
{
    let py = load_python(&format!("results/matrix/host_hilbert{N}_python.json"));
    let vals = &py["values"];
    let a = fill_matrix_json_n::<N>(&vals["A"]);
    let b = col_from_json::<N>(&vals["b"]);
    let x_gold = col_from_json::<N>(&vals["x_gold"]);

    let a_lu = fill_matrix_json_n::<N>(&vals["A"]);
    let lu = LuDecomposition::decompose(a_lu).expect("lu");
    let mut x = b;
    lu.solve_mut::<1>(&mut x).expect("solve");
    let residual_ratio = solve_residual_ratio(&a, &x, &b);
    let rel_err = rel_err_col(&x, &x_gold);

    write_artifact(
        PathBuf::from(format!("results/matrix/host_hilbert{N}_native.json")),
        &json!({
            "slug": "matrix",
            "source": "rust",
            "fixture": name,
            "values": {
                "x": col_to_json(&x),
                "residual_ratio": residual_ratio,
                "rel_err": rel_err,
            },
            "series": {},
        }),
    );
}

fn emit_poly52() {
    let py = load_python("results/polynomial/host_poly52_python.json");
    let vals = &py["values"];
    let coeffs_arr = vals["coeffs"].as_array().unwrap();
    let mut coeffs = [0.0_f64; 52];
    for (i, c) in coeffs_arr.iter().enumerate() {
        coeffs[i] = c.as_f64().unwrap();
    }
    let p = ArrayPolynomial::<f64, 52>::from_fn(|i| coeffs[i]);
    let poly_x = vals["poly_x"].as_f64().unwrap();
    let got = p.evaluate(poly_x);
    let poly_val = vals["poly_val"].as_f64().unwrap();
    let rel = if poly_val.abs() == 0.0 {
        (got - poly_val).abs()
    } else {
        (got - poly_val).abs() / poly_val.abs()
    };
    write_artifact(
        "results/polynomial/host_poly52_native.json",
        &json!({
            "slug": "polynomial",
            "source": "rust",
            "fixture": "poly52",
            "values": { "poly_val": got, "rel_err": rel },
            "series": {},
        }),
    );
}

fn emit_tf_clustered() {
    let py = load_python("results/transfer_function/host_tf_clustered_python.json");
    let vals = &py["values"];
    let den_arr = vals["den"].as_array().unwrap();
    let mut den = [0.0_f64; 52];
    for (i, c) in den_arr.iter().enumerate() {
        den[i] = c.as_f64().unwrap();
    }
    let tf = ArrayTransferFunction::<f64, 1, 52>::continuous([1.0], den);
    let omegas = vals["omegas"].as_array().unwrap();
    let h_re = vals["h_re"].as_array().unwrap();
    let h_im = vals["h_im"].as_array().unwrap();
    let mut rels = Vec::new();
    for i in 0..omegas.len() {
        let w = omegas[i].as_f64().unwrap();
        let h = tf.eval_frequency(w);
        let gre = h_re[i].as_f64().unwrap();
        let gim = h_im[i].as_f64().unwrap();
        let gabs = (gre * gre + gim * gim).sqrt();
        let diff = ((h.re - gre).hypot(h.im - gim)).abs();
        rels.push(if gabs == 0.0 { diff } else { diff / gabs });
    }
    write_artifact(
        "results/transfer_function/host_tf_clustered_native.json",
        &json!({
            "slug": "transfer_function",
            "source": "rust",
            "fixture": "tf_clustered",
            "values": { "rel_errs": rels },
            "series": {},
        }),
    );
}

fn inf_norm_mat<const N: usize>(m: &Owned<f64, N, N>) -> f64
where
    Const<N>: Dim,
{
    let mut best = 0.0_f64;
    for i in 0..N {
        let mut s = 0.0_f64;
        for j in 0..N {
            s += m.get(i, j).copied().unwrap().abs();
        }
        if s > best {
            best = s;
        }
    }
    best
}

fn emit_stiff_zoh() {
    let py = load_python("results/state_space/host_stiff_zoh_python.json");
    let vals = &py["values"];
    let a = fill_matrix_json::<2, 2>(&vals["A"]);
    let ad_gold = fill_matrix_json::<2, 2>(&vals["Ad_gold"]);
    let dt = vals["dt"].as_f64().unwrap();
    let b = Owned::<f64, 2, 1>::from_array([[0.0, 1.0]]);
    let c = Owned::<f64, 1, 2>::from_array([[1.0], [0.0]]);
    let d = Owned::<f64, 1, 1>::zero();
    let sys = ArrayStateSpace::continuous(a, b, c, d);
    let sys_d = sys.to_discrete_zoh(dt);
    let ad = sys_d.a();
    let rel = rel_err_mat(&ad, &ad_gold);
    let gold_norm = inf_norm_mat(&ad_gold);
    let den = gold_norm * f64::EPSILON;
    let residual_ratio = if den == 0.0 { 0.0 } else { rel * gold_norm / den };
    write_artifact(
        "results/state_space/host_stiff_zoh_native.json",
        &json!({
            "slug": "state_space",
            "source": "rust",
            "fixture": "stiff_zoh",
            "values": { "rel_err": rel, "residual_ratio": residual_ratio },
            "series": {},
        }),
    );
}

fn on_big_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .name("host-scale-1024".into())
        .stack_size(BIG_STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("join")
}

fn emit_gemm1024() {
    on_big_stack(|| {
        let py = load_python("results/matrix/host_gemm1024_python.json");
        let c_gold_arr = &py["values"]["C_gold"];
        let a = well_conditioned::<1024>(0.01);
        let b = well_conditioned::<1024>(0.02);
        let mut c = Owned::<f64, 1024, 1024>::zero();
        a.mul_into(&b, &mut c);
        let t0 = Instant::now();
        for _ in 0..ITERS_1024 {
            a.mul_into(&b, &mut c);
        }
        let measured = t0.elapsed().as_secs_f64() / f64::from(ITERS_1024);
        let mut c_gold = Owned::<f64, 1024, 1024>::zero();
        let rows = c_gold_arr.as_array().unwrap();
        for i in 0..1024 {
            let row = rows[i].as_array().unwrap();
            for j in 0..1024 {
                c_gold.set(i, j, row[j].as_f64().unwrap()).expect("set");
            }
        }
        let rel = rel_err_mat(&c, &c_gold);
        write_artifact(
            "results/matrix/host_gemm1024_native.json",
            &json!({
                "slug": "matrix",
                "source": "rust",
                "fixture": "gemm1024",
                "values": { "rel_err": rel, "measured_secs": measured },
                "series": {},
            }),
        );
    });
}

fn emit_lu1024() {
    on_big_stack(|| {
        let py = load_python("results/matrix/host_lu1024_python.json");
        let vals = &py["values"];
        let a_lu = well_conditioned::<1024>(0.01);
        let b_rhs = col_from_json::<1024>(&vals["b"]);
        let x_gold = col_from_json::<1024>(&vals["x_gold"]);
        let lu = LuDecomposition::decompose(a_lu).expect("lu");
        let mut x = b_rhs;
        lu.solve_mut::<1>(&mut x).expect("solve");
        let a_ref = well_conditioned::<1024>(0.01);
        let b_ref = col_from_json::<1024>(&vals["b"]);
        let residual_ratio = solve_residual_ratio(&a_ref, &x, &b_ref);
        let rel_x = rel_err_col(&x, &x_gold);
        write_artifact(
            "results/matrix/host_lu1024_native.json",
            &json!({
                "slug": "matrix",
                "source": "rust",
                "fixture": "lu1024",
                "values": {
                    "residual_ratio": residual_ratio,
                    "rel_err": rel_x,
                },
                "series": {},
            }),
        );
    });
}

fn emit_tensor1024() {
    on_big_stack(|| {
        let py = load_python("results/tensor/host_tensor1024_python.json");
        let vals = &py["values"];
        let points = vals["points"].as_array().unwrap();
        let gold = vals["samples_gold"].as_array().unwrap();
        let grid = ArrayTensor::<f32, 1024, 1024>::from_fn(|idx| (idx[0] + idx[1]) as f32);
        let mut got = Vec::new();
        for (pt, g) in points.iter().zip(gold.iter()) {
            let pt_arr = pt.as_array().unwrap();
            let sample = grid.interpolate(&[
                pt_arr[0].as_f64().unwrap() as f32,
                pt_arr[1].as_f64().unwrap() as f32,
            ]);
            got.push((sample, g.as_f64().unwrap() as f32));
        }
        let abs_errs: Vec<f32> = got.iter().map(|(s, g)| (s - g).abs()).collect();
        write_artifact(
            "results/tensor/host_tensor1024_native.json",
            &json!({
                "slug": "tensor",
                "source": "rust",
                "fixture": "tensor1024",
                "values": { "abs_errs": abs_errs },
                "series": {},
            }),
        );
    });
}

/// Emit all host-scale native JSON artifacts (requires Python host-scale JSON).
pub fn emit_all() {
    emit_hilbert12();
    emit_hilbert16();
    emit_hilbert128();
    emit_poly52();
    emit_tf_clustered();
    emit_stiff_zoh();
    emit_gemm1024();
    emit_lu1024();
    emit_tensor1024();
}
