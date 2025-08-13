// #![allow(dead_code)]
// use ark_ff::PrimeField;
// use ark_poly::DenseMultilinearExtension;
// use rayon::prelude::*;
// use regex::Regex;
// use std::collections::{HashMap, HashSet};
// use std::fmt;
// use std::sync::Arc;
// use ark_poly::{DenseMultilinearExtension, EvaluationDomain, GeneralEvaluationDomain};
// use super::data_structures::*;

// // --- Parser errors ---
// #[derive(Debug)]
// pub enum ParseError {
//     UnexpectedChar(char, usize),
//     UnexpectedEnd,
//     UnknownVar(String),
//     InvalidNumber(String),
//     UnbalancedParens,
//     Other(String),
// }
// impl fmt::Display for ParseError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         use ParseError::*;
//         match self {
//             UnexpectedChar(c, i) => write!(f, "Unexpected char '{}' at {}", c, i),
//             UnexpectedEnd => write!(f, "Unexpected end of input"),
//             UnknownVar(s) => write!(f, "Unknown variable '{}'", s),
//             InvalidNumber(s) => write!(f, "Invalid number '{}'", s),
//             UnbalancedParens => write!(f, "Unbalanced parentheses"),
//             Other(s) => write!(f, "{}", s),
//         }
//     }
// }
// impl std::error::Error for ParseError {}

// // --- Variable Bindings ---
// pub struct VariableBindings<F: PrimeField> {
//     pub index_map: HashMap<String, usize>,
//     pub mle_store: Vec<Arc<DenseMultilinearExtension<F>>>,
// }

// impl<F: PrimeField> VariableBindings<F> {
//     pub fn new() -> Self {
//         Self {
//             index_map: HashMap::new(),
//             mle_store: Vec::new(),
//         }
//     }

//     pub fn insert(&mut self, name: &str, mle: Arc<DenseMultilinearExtension<F>>) -> usize {
//         if let Some(&idx) = self.index_map.get(name) {
//             idx
//         } else {
//             let idx = self.mle_store.len();
//             self.mle_store.push(mle);
//             self.index_map.insert(name.to_string(), idx);
//             idx
//         }
//     }

//     pub fn get_index(&self, name: &str) -> Option<usize> {
//         self.index_map.get(name).copied()
//     }

//     pub fn set_by_index(&mut self, idx: usize, mle: Arc<DenseMultilinearExtension<F>>) {
//         self.mle_store[idx] = mle;
//     }
// }

// // --- AST ---
// #[derive(Debug, Clone)]
// enum Node {
//     Var(String),
//     Const(i64),
//     Add(Box<Node>, Box<Node>),
//     Sub(Box<Node>, Box<Node>),
//     Mul(Box<Node>, Box<Node>),
//     Pow(Box<Node>, u32),
// }

// // --- Parser ---
// struct Parser<'a> {
//     chars: Vec<char>,
//     pos: usize,
//     _inp: &'a str,
// }

// impl<'a> Parser<'a> {
//     fn new(s: &'a str) -> Self {
//         Self { chars: s.chars().collect(), pos: 0, _inp: s }
//     }
//     fn peek(&self) -> Option<char> { self.chars.get(self.pos).copied() }
//     fn bump(&mut self) -> Option<char> { let c = self.peek(); if c.is_some() { self.pos += 1; } c }
//     fn eat_ws(&mut self) { while matches!(self.peek(), Some(c) if c.is_whitespace()) { self.pos += 1; } }

//     fn parse(&mut self) -> Result<Node, ParseError> {
//         self.eat_ws();
//         let n = self.parse_expr()?;
//         self.eat_ws();
//         if self.pos < self.chars.len() { return Err(ParseError::UnexpectedChar(self.chars[self.pos], self.pos)); }
//         Ok(n)
//     }

//     fn parse_expr(&mut self) -> Result<Node, ParseError> {
//         let mut node = self.parse_term()?;
//         loop {
//             self.eat_ws();
//             match self.peek() {
//                 Some('+') => { self.bump(); let rhs = self.parse_term()?; node = Node::Add(Box::new(node), Box::new(rhs)); }
//                 Some('-') => { self.bump(); let rhs = self.parse_term()?; node = Node::Sub(Box::new(node), Box::new(rhs)); }
//                 _ => break,
//             }
//         }
//         Ok(node)
//     }

//     fn parse_term(&mut self) -> Result<Node, ParseError> {
//         let mut node = self.parse_factor()?;
//         loop {
//             self.eat_ws();
//             if let Some('*') = self.peek() {
//                 if self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '*' { break; }
//                 else { self.bump(); let rhs = self.parse_factor()?; node = Node::Mul(Box::new(node), Box::new(rhs)); }
//             } else { break; }
//         }
//         Ok(node)
//     }

//     fn parse_factor(&mut self) -> Result<Node, ParseError> {
//         self.eat_ws();
//         match self.peek() {
//             Some('(') => { self.bump(); let inner = self.parse_expr()?; self.eat_ws(); if self.peek() == Some(')') { self.bump(); self.parse_pow_suffix(inner) } else { Err(ParseError::UnbalancedParens) } }
//             Some(c) if is_ident_start(c) => { let id = self.parse_ident(); self.parse_pow_suffix(Node::Var(id)) }
//             Some(c) if c.is_ascii_digit() => { let n = self.parse_number()?; self.parse_pow_suffix(Node::Const(n)) }
//             Some('-') => { self.bump(); let f = self.parse_factor()?; Ok(Node::Sub(Box::new(Node::Const(0)), Box::new(f))) }
//             Some(c) => Err(ParseError::UnexpectedChar(c, self.pos)),
//             None => Err(ParseError::UnexpectedEnd),
//         }
//     }

//     fn parse_pow_suffix(&mut self, base: Node) -> Result<Node, ParseError> {
//         self.eat_ws();
//         if let Some('^') = self.peek() { self.bump(); let e = self.parse_unsigned_integer()?; Ok(Node::Pow(Box::new(base), e as u32)) }
//         else if let Some('*') = self.peek() { if self.pos + 1 < self.chars.len() && self.chars[self.pos + 1] == '*' { self.pos += 2; let e = self.parse_unsigned_integer()?; Ok(Node::Pow(Box::new(base), e as u32)) } else { Ok(base) } }
//         else { Ok(base) }
//     }

//     fn parse_ident(&mut self) -> String {
//         let mut s = String::new();
//         if let Some(c) = self.peek() { if is_ident_start(c) { s.push(c); self.bump(); } }
//         while let Some(c) = self.peek() { if is_ident_continue(c) { s.push(c); self.bump(); } else { break; } }
//         s
//     }

//     fn parse_number(&mut self) -> Result<i64, ParseError> {
//         let mut s = String::new();
//         while let Some(c) = self.peek() { if c.is_ascii_digit() { s.push(c); self.bump(); } else { break; } }
//         s.parse::<i64>().map_err(|_| ParseError::InvalidNumber(s))
//     }

//     fn parse_unsigned_integer(&mut self) -> Result<u64, ParseError> {
//         let mut s = String::new();
//         while let Some(c) = self.peek() { if c.is_ascii_digit() { s.push(c); self.bump(); } else { break; } }
//         if s.is_empty() { return Err(ParseError::Other("expected integer".into())); }
//         s.parse::<u64>().map_err(|_| ParseError::Other(format!("invalid {}", s)))
//     }
// }

// fn is_ident_start(c: char) -> bool { c.is_ascii_alphabetic() || c == '_' }
// fn is_ident_continue(c: char) -> bool { c.is_ascii_alphanumeric() || c == '_' }

// // --- Extract variable names ---
// pub fn extract_variable_names(input: &str) -> Vec<String> {
//     let re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
//     let mut seen = HashSet::new();
//     let mut vars = Vec::new();
//     for cap in re.captures_iter(input) {
//         let name = cap.get(0).unwrap().as_str().to_string();
//         if name.chars().all(|c| c.is_ascii_digit()) { continue; }
//         if seen.insert(name.clone()) { vars.push(name); }
//     }
//     vars
// }

// // --- AST → Product Lists ---
// type ProductArc<F> = (F, Vec<Arc<DenseMultilinearExtension<F>>>);

// fn ast_to_products<F>(
//     node: &Node,
//     var_map: &HashMap<String, Arc<DenseMultilinearExtension<F>>>,
//     const_to_mle: &dyn Fn(i64) -> Arc<DenseMultilinearExtension<F>>,
//     int_to_field: &dyn Fn(i64) -> F,
// ) -> Result<Vec<ProductArc<F>>, ParseError>
// where
//     F: PrimeField + Clone,
// {
//     match node {
//         Node::Var(name) => {
//             let mle = var_map.get(name).ok_or_else(|| ParseError::UnknownVar(name.clone()))?;
//             Ok(vec![(int_to_field(1), vec![Arc::clone(mle)])])
//         }
//         Node::Const(n) => Ok(vec![(int_to_field(1), vec![const_to_mle(*n)])]),
//         Node::Add(l, r) => {
//             let mut left = ast_to_products(l, var_map, const_to_mle, int_to_field)?;
//             let right = ast_to_products(r, var_map, const_to_mle, int_to_field)?;
//             left.extend(right.into_iter());
//             Ok(left)
//         }
//         Node::Sub(l, r) => {
//             let mut left = ast_to_products(l, var_map, const_to_mle, int_to_field)?;
//             let mut right = ast_to_products(r, var_map, const_to_mle, int_to_field)?;
//             let minus_one = int_to_field(-1);
//             for (coef, _refs) in right.iter_mut() { *coef *= minus_one.clone(); }
//             left.extend(right.into_iter());
//             Ok(left)
//         }
//         Node::Mul(l, r) => {
//             let left = ast_to_products(l, var_map, const_to_mle, int_to_field)?;
//             let right = ast_to_products(r, var_map, const_to_mle, int_to_field)?;
//             let mut out = Vec::with_capacity(left.len() * right.len());
//             for (ca, ra) in left.iter() { for (cb, rb) in right.iter() {
//                 let mut refs = Vec::with_capacity(ra.len() + rb.len());
//                 refs.extend(ra.iter().cloned());
//                 refs.extend(rb.iter().cloned());
//                 out.push((ca.clone() * cb.clone(), refs));
//             } }
//             Ok(out)
//         }
//         Node::Pow(base, exp) => {
//             let mut acc: Option<Vec<ProductArc<F>>> = None;
//             let mut base_prods = ast_to_products(base, var_map, const_to_mle, int_to_field)?;
//             let mut e = *exp;
//             while e > 0 {
//                 if (e & 1) == 1 {
//                     acc = Some(match acc { None => base_prods.clone(), Some(a) => mul_product_lists(&a, &base_prods), });
//                 }
//                 e >>= 1;
//                 if e > 0 { base_prods = mul_product_lists(&base_prods, &base_prods); }
//             }
//             Ok(acc.unwrap_or_else(|| vec![]))
//         }
//     }
// }

// fn mul_product_lists<F>(a: &Vec<ProductArc<F>>, b: &Vec<ProductArc<F>>) -> Vec<ProductArc<F>>
// where F: PrimeField + Clone {
//     let mut res = Vec::with_capacity(a.len() * b.len());
//     for (ca, ra) in a.iter() { for (cb, rb) in b.iter() {
//         let mut refs = Vec::with_capacity(ra.len() + rb.len());
//         refs.extend(ra.iter().cloned());
//         refs.extend(rb.iter().cloned());
//         res.push((ca.clone() * cb.clone(), refs));
//     } }
//     res
// }

// // --- Public entry point ---
// pub fn parse_to_virtual_polynomial<F>(
//     input: &str,
//     var_map: &HashMap<String, Arc<DenseMultilinearExtension<F>>>,
//     const_to_mle: &dyn Fn(i64) -> Arc<DenseMultilinearExtension<F>>,
//     int_to_field: &dyn Fn(i64) -> F,
//     nv: usize,
// ) -> Result<(VirtualPolynomial<F>, VariableBindings<F>), ParseError>
// where
//     F: PrimeField + Clone,
// {
//     let mut parser = Parser::new(input);
//     let ast = parser.parse()?;

//     let mut bindings = VariableBindings::new();
//     let mut vp = VirtualPolynomial::new(nv);

//     let products = ast_to_products(&ast, var_map, const_to_mle, int_to_field)?;

//     for (coef, refs) in products {
//         let indices: Vec<usize> = refs.into_iter().map(|mle| bindings.insert("<anon>", mle)).collect();
//         vp.add_product_by_indices(&indices, coef);
//     }

//     Ok((vp, bindings))
// }

// pub fn prepare_virtual_polynomial_with_zeroing<F>(
//     input: &str,
//     degree: usize,
//     pool: &rayon::ThreadPool,
// ) -> Result<(VirtualPolynomial<F>, VariableBindings<F>), ParseError>
// where
//     F: PrimeField + Clone,
// {
//     // Create evaluation domain
//     let domain = GeneralEvaluationDomain::<F>::new(degree).unwrap();

//     // Factory to convert integer constants to MLEs
//     let const_factory = |n: i64| {
//         let vals: Vec<F> = (0..degree).map(|_| F::from(n as u64)).collect();
//         Arc::new(DenseMultilinearExtension::from_evaluations_vec(vals))
//     };

//     let int_to_field = |n: i64| {
//         if n >= 0 { F::from(n as u64) } else { -F::from((-n) as u64) }
//     };

//     // Extract variable names
//     let variable_names = extract_variable_names(input);

//     // Generate random MLEs for variables
//     let mut var_map: HashMap<String, Arc<DenseMultilinearExtension<F>>> = HashMap::new();
//     for var in variable_names.iter() {
//         let vals: Vec<F> = pool.install(|| {
//             (0..degree)
//                 .into_par_iter()
//                 .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
//                 .collect()
//         });
//         let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(vals));
//         var_map.insert(var.clone(), mle);
//     }

//     // Parse input into VirtualPolynomial
//     let (mut vp, mut bindings) =
//         parse_to_virtual_polynomial::<F>(input, &var_map, &const_factory, &int_to_field, variable_names.len())?;

//     // Evaluate expression at all domain points
//     let eval_vals: Vec<F> = domain.elements().map(|pt| vp.evaluate_at_point(pt)).collect();

//     // Create zeroizing variable "o" as DenseMultilinearExtension
//     let o_mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(eval_vals));

//     // Add o to VariableBindings
//     let o_idx = bindings.insert("o", Arc::clone(&o_mle));

//     // Add -1 * o product to VirtualPolynomial
//     vp.add_product_by_indices(&[o_idx], int_to_field(-1));

//     Ok((vp, bindings))
// }