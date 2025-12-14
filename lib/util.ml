(* contains useful functions *)

let make_rng seed =
  Random.init seed;
  ()

let print_array arr =
  Array.iter (fun x -> Printf.printf "%.4f" x) arr;
  print_endline ""

(* function that takes an integer n for number of data points to generate *)
let make_sin_data n =
  let data = ref [] in
  for _ = 1 to n do
    let x = (Random.float 10.) -. 5. in
    let y = Float.sin x +. 0.3 *. (Random.float 1.0 -. 0.5) in
    data := ([| x |], [| y |]) :: !data
  done;
  !data

(* Generate polynomial curve dataset: y = 2*x^2 + x + 1 + noise
   Generates n data points in range [-5, 5] with noise matching Python's scale *)
let make_polynomial_data n =
  let data = ref [] in
  for _ = 1 to n do
    let x = (Random.float 20.0) -. 10.0 in  (* x in [-5, 5] *)
    (* Polynomial: y = 2*x^2 + x + 1 *)
    let y_true = -2.0 *. (x ** 2.0) +. 5. *. x +. 10.0 in
    (* noise ~ U[-1.25, 1.25] to match Python's N(0, 1.25^2) scale *)
    let noise = 1.25 *. (2.0 *. Random.float 1.0 -. 1.0) in
    let y = y_true +. noise in
    data := ([| x |], [| y |]) :: !data
  done;
  !data

(* Generate linear regression dataset: y = 2.5*x - 1.5 + noise
   Generates n data points in range [-5, 5] with Gaussian noise *)
let make_linear_data n =
  let data = ref [] in
  for _ = 1 to n do
    let x = (Random.float 20.0) -. 10.0 in  (* x in [-5, 5] *)
    (* Linear: y = 2.5*x - 1.5 *)
    let y_true = 10. *. x -. 1.5 in
    let noise = 4.0 *. (Random.float 1.0 -. 0.5) in  (* noise in [-0.2, 0.2] *)
    let y = y_true +. noise in
    data := ([| x |], [| y |]) :: !data
  done;
  !data