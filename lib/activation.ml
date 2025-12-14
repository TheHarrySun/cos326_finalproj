(* stores activation functions and their derivatives *)

type activation = Tanh | Relu | Identity

let apply activation x =
  match activation with
  | Tanh -> Float.tanh x
  | Relu -> Float.max 0.0 x
  | Identity -> x

let deriv activation pre post =
  match activation with
  | Tanh -> 1.0 -. (post *. post)
  | Relu -> if pre > 0.0 then 1.0 else 0.
  | Identity -> 1.0