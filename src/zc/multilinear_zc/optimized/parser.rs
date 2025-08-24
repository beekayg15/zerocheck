#![allow(dead_code)]
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain,
};
use rayon::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use super::data_structures::*;

// use crate::your_module::ParseError as InternalParseError; // optional

// --- Parser errors (local) ---
#[derive(Debug)]
pub enum ParseError {
    UnexpectedChar(char, usize),
    UnexpectedEnd,
    UnknownVar(String),
    InvalidNumber(String),
    UnbalancedParens,
    Other(String),
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParseError::*;
        match self {
            UnexpectedChar(c, i) => write!(f, "Unexpected char '{}' at {}", c, i),
            UnexpectedEnd => write!(f, "Unexpected end of input"),
            UnknownVar(s) => write!(f, "Unknown variable '{}'", s),
            InvalidNumber(s) => write!(f, "Invalid number '{}'", s),
            UnbalancedParens => write!(f, "Unbalanced parentheses"),
            Other(s) => write!(f, "{}", s),
        }
    }
}
impl std::error::Error for ParseError {}

// --- AST ---
#[derive(Debug, Clone)]
enum Node {
    Var(String),
    Const(i64),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Pow(Box<Node>, u32),
}

// --- Parser (very similar to previous prototype) ---
struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    _inp: &'a str,
}
impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
            _inp: s,
        }
    }
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn eat_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn parse(&mut self) -> Result<Node, ParseError> {
        self.eat_ws();
        let n = self.parse_expr()?;
        self.eat_ws();
        if self.pos < self.chars.len() {
            return Err(ParseError::UnexpectedChar(self.chars[self.pos], self.pos));
        }
        Ok(n)
    }

    fn parse_expr(&mut self) -> Result<Node, ParseError> {
        let mut node = self.parse_term()?;
        loop {
            self.eat_ws();
            match self.peek() {
                Some('+') => {
                    self.bump();
                    let rhs = self.parse_term()?;
                    node = Node::Add(Box::new(node), Box::new(rhs));
                }
                Some('-') => {
                    self.bump();
                    let rhs = self.parse_term()?;
                    node = Node::Sub(Box::new(node), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_term(&mut self) -> Result<Node, ParseError> {
        let mut node = self.parse_factor()?;
        loop {
            self.eat_ws();
            // '*' for multiply, but treat '**' (power) specially in factor
            if let Some('*') = self.peek() {
                if self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '*' {
                    // power operator handled in factor -> break
                    break;
                } else {
                    self.bump();
                    let rhs = self.parse_factor()?;
                    node = Node::Mul(Box::new(node), Box::new(rhs));
                }
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_factor(&mut self) -> Result<Node, ParseError> {
        self.eat_ws();
        match self.peek() {
            Some('(') => {
                self.bump();
                let inner = self.parse_expr()?;
                self.eat_ws();
                if self.peek() == Some(')') {
                    self.bump();
                    self.parse_pow_suffix(inner)
                } else {
                    Err(ParseError::UnbalancedParens)
                }
            }
            Some(c) if is_ident_start(c) => {
                let id = self.parse_ident();
                self.parse_pow_suffix(Node::Var(id))
            }
            Some(c) if c.is_ascii_digit() => {
                let n = self.parse_number()?;
                self.parse_pow_suffix(Node::Const(n))
            }
            Some('-') => {
                self.bump();
                // unary minus -> 0 - factor
                let f = self.parse_factor()?;
                Ok(Node::Sub(Box::new(Node::Const(0)), Box::new(f)))
            }
            Some(c) => Err(ParseError::UnexpectedChar(c, self.pos)),
            None => Err(ParseError::UnexpectedEnd),
        }
    }

    fn parse_pow_suffix(&mut self, base: Node) -> Result<Node, ParseError> {
        self.eat_ws();
        if let Some('^') = self.peek() {
            self.bump();
            let e = self.parse_unsigned_integer()?;
            Ok(Node::Pow(Box::new(base), e as u32))
        } else if let Some('*') = self.peek() {
            if self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '*' {
                self.pos += 2;
                let e = self.parse_unsigned_integer()?;
                Ok(Node::Pow(Box::new(base), e as u32))
            } else {
                Ok(base)
            }
        } else {
            Ok(base)
        }
    }

    fn parse_ident(&mut self) -> String {
        let mut s = String::new();
        if let Some(c) = self.peek() {
            if is_ident_start(c) {
                s.push(c);
                self.bump();
            }
        }
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        s
    }

    fn parse_number(&mut self) -> Result<i64, ParseError> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        s.parse::<i64>().map_err(|_| ParseError::InvalidNumber(s))
    }

    fn parse_unsigned_integer(&mut self) -> Result<u64, ParseError> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if s.is_empty() {
            return Err(ParseError::Other("expected integer".into()));
        }
        s.parse::<u64>()
            .map_err(|_| ParseError::Other(format!("invalid {}", s)))
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// --- Convert AST to intermediate product-list ---
// ProductArc = (coefficient: F, refs: Vec<Arc<DenseMultilinearExtension<F>>>)
type ProductArc<F> = (F, Vec<Arc<DenseMultilinearExtension<F>>>);

fn ast_to_products<F>(
    node: &Node,
    var_map: &HashMap<String, Arc<DenseMultilinearExtension<F>>>,
    const_to_eval: &dyn Fn(i64) -> Arc<DenseMultilinearExtension<F>>,
    int_to_field: &dyn Fn(i64) -> F,
) -> Result<Vec<ProductArc<F>>, ParseError>
where
    F: PrimeField + Clone,
{
    match node {
        Node::Var(name) => {
            let arc = var_map
                .get(name)
                .ok_or_else(|| ParseError::UnknownVar(name.clone()))?;
            Ok(vec![(int_to_field(1), vec![Arc::clone(arc)])])
        }

        Node::Const(n) => {
            Ok(vec![(int_to_field(*n), vec![])]) // coefficient only, no evaluation
        }

        Node::Add(l, r) => {
            let mut left = ast_to_products::<F>(l, var_map, const_to_eval, int_to_field)?;
            let right = ast_to_products::<F>(r, var_map, const_to_eval, int_to_field)?;
            left.extend(right.into_iter());
            Ok(left)
        }
        Node::Sub(l, r) => {
            let mut left = ast_to_products::<F>(l, var_map, const_to_eval, int_to_field)?;
            let mut right = ast_to_products::<F>(r, var_map, const_to_eval, int_to_field)?;
            // negate right coefficients
            let minus_one = int_to_field(-1);
            for (coef, _refs) in right.iter_mut() {
                *coef = coef.clone() * minus_one.clone();
            }
            left.extend(right.into_iter());
            Ok(left)
        }
        Node::Mul(l, r) => {
            let left = ast_to_products::<F>(l, var_map, const_to_eval, int_to_field)?;
            let right = ast_to_products::<F>(r, var_map, const_to_eval, int_to_field)?;
            let mut out = Vec::with_capacity(left.len() * right.len());
            for (ca, ra) in left.iter() {
                for (cb, rb) in right.iter() {
                    let mut refs = Vec::with_capacity(ra.len() + rb.len());
                    refs.extend(ra.iter().cloned());
                    refs.extend(rb.iter().cloned());
                    let coef = ca.clone() * cb.clone();
                    out.push((coef, refs));
                }
            }
            Ok(out)
        }
        Node::Pow(base, exp) => {
            let mut acc: Option<Vec<ProductArc<F>>> = None;
            let mut base_prods = ast_to_products::<F>(base, var_map, const_to_eval, int_to_field)?;
            let mut e = *exp;
            while e > 0 {
                if (e & 1) == 1 {
                    acc = Some(match acc {
                        None => base_prods.clone(),
                        Some(a) => mul_product_lists::<F>(&a, &base_prods),
                    });
                }
                e >>= 1;
                if e > 0 {
                    base_prods = mul_product_lists::<F>(&base_prods, &base_prods);
                }
            }
            Ok(acc.unwrap_or_else(|| vec![]))
        }
    }
}

/// Multiply two product-lists: cross product and combine refs/coefs
fn mul_product_lists<F: PrimeField>(
    left: &Vec<(F, Vec<Arc<DenseMultilinearExtension<F>>>)>,
    right: &Vec<(F, Vec<Arc<DenseMultilinearExtension<F>>>)>,
) -> Vec<(F, Vec<Arc<DenseMultilinearExtension<F>>>)> {
    let mut result = Vec::new();
    for (lc, lrefs) in left {
        for (rc, rrefs) in right {
            let mut new_refs = lrefs.clone();
            new_refs.extend(rrefs.iter().cloned());
            result.push((*lc * *rc, new_refs));
        }
    }
    result
}

// --- Public entry point ---
/// Parse `input` into `VirtualPolynomial<F>`.
///
/// - `var_map` maps variable names to precomputed `Arc<DenseMultilinearExtension<F>>`.
/// - `const_to_eval` should produce an `Arc<DenseMultilinearExtension<F>>` that is the constant vector equal to the input integer on the domain.
/// - `int_to_field` converts small integers into `F` (used for negation and internal coefficients).
pub fn parse_to_virtual_evaluation<F>(
    input: &str,
    var_map: &HashMap<String, Arc<DenseMultilinearExtension<F>>>,
    const_to_eval: &dyn Fn(i64) -> Arc<DenseMultilinearExtension<F>>,
    int_to_field: &dyn Fn(i64) -> F,
    nv: usize,
) -> Result<VirtualPolynomial<F>, ParseError>
where
    F: PrimeField + Clone,
{
    let mut parser = Parser::new(input);
    let ast = parser.parse()?;
    let products = ast_to_products::<F>(&ast, var_map, const_to_eval, int_to_field)?;

    // assemble into real VirtualPolynomial<F>
    let mut ve = VirtualPolynomial::new(nv);

    for (coef, refs) in products.into_iter() {
        // refs: Vec<Arc<DenseMultilinearExtension<F>>>; add_product will deduplicate pointers internally
        ve.add_product(refs.into_iter(), coef);
    }

    Ok(ve)
}

// Extract variable names from AST
pub fn extract_variable_names(input: &str) -> Vec<String> {
    // Matches identifiers starting with a letter or underscore, then alphanumeric/underscore
    // This will match q1, my_var, _temp, etc.
    let re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
    let mut seen = HashSet::new();
    let mut vars = Vec::new();

    for cap in re.captures_iter(input) {
        let name = cap.get(0).unwrap().as_str().to_string();
        // Skip things that look like numbers (defensive)
        if name.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        // Avoid duplicates while preserving order
        if seen.insert(name.clone()) {
            vars.push(name);
        }
    }

    vars
}

// Generate virtual evaluation from a string input.
// Rule for input:
// - Variables: alphanumeric names starting with a letter (e.g., `x`, `var1`)
// - Constants: integers (e.g., `42`, `-3`)
// - Operations: `+`, `-`, `*`, `^` (power), parentheses for grouping (terms have to be connected by operators)
// e.g. `x + 2*y - 3*z^2 + (4 - x)`
pub fn prepare_virtual_evaluation_from_string<F>(
    input: &str,
    degree: usize,
    nv: usize,
) -> Result<VirtualPolynomial<F>, ParseError>
where
    F: PrimeField + Clone,
{
    // Factory to create constant evaluations on the domain
    let const_factory = |n: i64| {
        let vals: Vec<F> = (0..degree).map(|_| F::from(n as u64)).collect();
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vals))
    };
    // let int_to_field = |n: i64| F::from((n as i128) as i64 as u64);
    let int_to_field = |n: i64| {
        if n >= 0 {
            F::from(n as u64)
        } else {
            -F::from((-n) as u64)
        }
    };

    // parse
    let variable_names = extract_variable_names(input);

    // Prepare variable map: variable names -> random evaluations
    let mut var_map: HashMap<String, Arc<DenseMultilinearExtension<F>>> = HashMap::new();

    // Randomly gernerate evals for variables
    for var in variable_names.into_iter() {
        let rand_evals: Vec<F> = (0..(1 << nv))
            .into_par_iter()
            .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
            .collect();
        let arc = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
            nv, rand_evals,
        ));
        var_map.insert(var, arc);
    }
    let ve = parse_to_virtual_evaluation::<F>(input, &var_map, &const_factory, &int_to_field, nv)
        .expect("parse ok");

    return Ok(ve);
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ark_bls12_381::Fr;
//     use ark_poly::domain::GeneralEvaluationDomain;
//     use ark_poly::DenseMultilinearExtension;
//     use std::sync::Arc;

//     // NOTE: test uses small domain and simple const->eval factory
//     #[test]
//     fn parse_into_virtual_eval_smoke() {
//         let degree = 2usize;
//         let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();

//         // prepare var_map: g,h,s -> random evals (here deterministic small vectors)
//         let g_vals: Vec<Fr> = (0..degree).map(|i| Fr::from(i as u64 + 1u64)).collect();
//         let h_vals: Vec<Fr> = (0..degree).map(|i| Fr::from(i as u64 + 2u64)).collect();
//         let s_vals: Vec<Fr> = (0..degree).map(|i| Fr::from(i as u64 + 3u64)).collect();

//         let g_arc = Arc::new(DenseMultilinearExtension::from_vec_and_domain(
//             g_vals, domain,
//         ));
//         let h_arc = Arc::new(DenseMultilinearExtension::from_vec_and_domain(
//             h_vals,
//             g_arc.domain().clone(),
//         ));
//         let s_arc = Arc::new(DenseMultilinearExtension::from_vec_and_domain(
//             s_vals,
//             g_arc.domain().clone(),
//         ));

//         let mut var_map: HashMap<String, Arc<DenseMultilinearExtension<Fr>>> = HashMap::new();
//         var_map.insert("g".to_string(), g_arc.clone());
//         var_map.insert("h".to_string(), h_arc.clone());
//         var_map.insert("s".to_string(), s_arc.clone());

//         // const -> eval factory (create constant vector on same domain)
//         let const_factory = |n: i64| {
//             let vals: Vec<Fr> = (0..degree).map(|_| Fr::from(n as u64)).collect();
//             Arc::new(DenseMultilinearExtension::from_vec_and_domain(
//                 vals,
//                 g_arc.domain().clone(),
//             ))
//         };
//         let int_to_field = |n: i64| Fr::from((n as i128) as i64 as u64);

//         // parse
//         let expr = "g*h*(1-g)";
//         let ve = parse_to_virtual_evaluation::<Fr>(expr, &var_map, &const_factory, &int_to_field)
//             .expect("parse ok");

//         print!("{:#?}", ve);
//         // Basic sanity: products should be non-empty
//         assert!(ve.products.len() > 0);
//         // max_multiplicand should be set >= 1
//         assert!(ve.evals_info.max_multiplicand == 3);
//     }
// }
