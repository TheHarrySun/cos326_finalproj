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
    let y = Float.sin x +. 0.3 *. (Random.float 1.0) in
    data := ([| x |], y) :: !data
  done;
  !data