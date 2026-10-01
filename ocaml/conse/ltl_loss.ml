open Isatensor;;
open Angstrom;;

(* Non-line whitespace: true for space and tab *)
let is_space = function
  | '\x20' | '\x09' -> true
  | _ -> false

(* Skip all spaces *)
let skip_space = skip_while is_space

(* Transform parser by skipping all trailing space *)
let lexeme p = p <* skip_space

(* Digits: true for 0 to 9 *)
let is_digit = function
  | '0' .. '9' -> true
  | _ -> false

(* Natural number: sequence of digits with result cast into an formal nat *)
let natural = take_while1 is_digit >>| (fun x -> Lib.Arith.nat_of_integer (Z.of_string x))

(* Comparison operators the parser should support *)
let comp_op =
      string "<="
  <|> string "<"
  <|> string "=="
  <|> string "!="
  <|> string ">="
  <|> string ">"

(* Comparison expression: natural, operator, natural *)
let comp_expr =
  lexeme natural >>= fun x ->
    lexeme comp_op >>= fun op ->
      lexeme natural >>| fun y ->
        match op with
        | "<=" -> Lib.LTLLoss.Lequal (x, y)
        | "<" -> Lib.LTLLoss.Less (x, y)
        | "==" -> Lib.LTLLoss.Equal (x, y)
        | "!=" -> Lib.LTLLoss.Nequal (x, y)
        | ">=" -> Lib.LTLLoss.Lequal (y, x)
        | ">" -> Lib.LTLLoss.Less (y, x)
        | _ -> failwith "Parsed but unrecognised comparison operation"

(* Inspiration for the following:
  https://hackage.haskell.org/package/parser-combinators-1.3.0/docs/src/Control.Monad.Combinators.Expr.html#makeExprParser
 *)

type 'a operator =
    Prefix of ('a -> 'a) t
  | InfixL of ('a -> 'a -> 'a) t
  | InfixR of ('a -> 'a -> 'a) t

type 'a batch = ('a -> 'a) t list * ('a -> 'a -> 'a) t list * ('a -> 'a -> 'a) t list

let splitOp (op : 'a operator) (b : 'a batch) =
  match b with
  | (prefixes, infixesL, infixesR) -> match op with
    | Prefix p -> (p :: prefixes, infixesL, infixesR)
    | InfixL p -> (prefixes, p :: infixesL, infixesR)
    | InfixR p -> (prefixes, infixesL, p :: infixesR)

let ident x = x

let pTerm (prefix : ('a -> 'a) t) (term: 'a t) : 'a t =
  (prefix <|> return ident) >>= fun f -> term >>| fun x -> f x

let rec pInfixL (op : ('a -> 'a -> 'a) t) (p : 'a t) (x : 'a) : 'a t =
  op >>= fun f -> p >>= fun y -> let r = f x y in pInfixL op p r <|> return r

let rec pInfixR (op : ('a -> 'a -> 'a) t) (p : 'a t) (x : 'a) : 'a t =
  op >>= fun f -> (p >>= fun r -> pInfixR op p r <|> return r) >>| fun y -> f x y

let addPrecLevel (term : 'a t) (ops : 'a operator list) : 'a t =
  let (prefixes, infixesL, infixesR) = List.fold_right splitOp ops ([], [], []) in
  let term' = pTerm (choice prefixes) term in
  let ras = pInfixR (choice infixesR) term' in
  let las = pInfixL (choice infixesL) term'
  in
    term' >>= fun x -> choice [ras x; las x; return x]

let makeExprParser (term : 'a t) (ops : 'a operator list list) : 'a t =
  List.fold_left addPrecLevel term ops

let ltl_op s f = lexeme (string s) *> return f
let prefix s f = Prefix (ltl_op s f)
let binaryL s f = InfixL (ltl_op s f)
let binaryR s f = InfixR (ltl_op s f)

let ltlImp x y = Lib.LTLLoss.Or (Lib.LTLLoss.nota x, y)
let ltlIff x y = Lib.LTLLoss.And (ltlImp x y, ltlImp y x)
let ltlTable = [ [ prefix "!" Lib.LTLLoss.nota
                 ; prefix "N" (fun x -> Lib.LTLLoss.Next x)
                 ; prefix "X" (fun x -> Lib.LTLLoss.WeakNext x)
                 ; prefix "G" (fun x -> Lib.LTLLoss.Always x)
                 ; prefix "F" (fun x -> Lib.LTLLoss.Eventually x)
                 ]
               ; [ binaryL "U" (fun x y -> Lib.LTLLoss.Until (x, y))
                 ; binaryL "R" (fun x y -> Lib.LTLLoss.Release (x, y))
                 ]
               ; [ binaryR "&" (fun x y -> Lib.LTLLoss.And (x, y))
                 ]
               ; [ binaryR "|" (fun x y -> Lib.LTLLoss.Or (x, y))
                 ]
               ; [ binaryR "<->" ltlIff
                 ]
               ; [ binaryL "->" ltlImp
                 ]
               ]

let between op cl p = lexeme op *> p <* lexeme cl
let parens p = between (char '(') (char ')') p

(* Constraint expression *)
let constr_expr : Lib.LTLLoss.constrainta t =
  fix (fun expr ->
    let term = parens expr <|> (comp_expr >>| fun x -> Lib.LTLLoss.Comp x) in
    makeExprParser term ltlTable
  )

(* Parser for Haskell-style lists given an element parser *)
let hs_list (e : 'a t) : 'a list t =
  let rec rest (x : unit) = (* Argument needed for rec to work *)
    (lexeme (char ',') *> e >>= fun x -> rest () >>= fun xs -> return (x :: xs)) <|> return [] in
  let elems = (e >>= fun x -> rest () >>= fun xs -> return (x :: xs)) <|> return [] in
  between (char '[') (char ']') elems

(* Dimensions are a list of naturals *)
let dims_expr = hs_list (lexeme natural)

(* Check the system is little-endian *)
(* TODO add support for big-endian systems instead of crashing *)
let () = if Sys.big_endian then failwith "Big-endian systems are currently not supported" else ()

(* Input and output of floats as byte onto a stream in little-endian IEE-754 *)
(* See: https://stackoverflow.com/a/5042153 *)
let output_float_le otch fv =
  let bits = ref (Int64.bits_of_float fv) in
  for _ = 0 to 7 do
    let byte = Int64.to_int (Int64.logand !bits 0xffL) in
    bits := Int64.shift_right_logical !bits 8;
    output_byte otch byte
  done
let rec output_all_float_le xs otch =
  match xs with
  | [] -> ()
  | (x::xs) ->
      let _ = output_float_le otch x in
      output_all_float_le xs otch
let input_float_le inch =
  let bits = ref 0L in
  for i = 0 to 7 do
    let byte = input_byte inch in
    bits := Int64.logor !bits (Int64.shift_left (Int64.of_int byte) (8 * i))
  done;
  Int64.float_of_bits !bits
let input_float_le_opt inch = try Some (input_float_le inch) with End_of_file -> None
let rec input_all_float_le inch =
  match (input_float_le_opt inch) with
  | None -> []
  | (Some x) -> x :: (input_all_float_le inch)

(* Elements input is all of STDIN *)
let vec = input_all_float_le In_channel.stdin

(* Computation mode: loss, gradient or both *)
type mode = Loss | Grad | Both

(* Read mode, gamma, constraint and shape from arguments *)
let (mode, gamma, constr, dims) =
  if Array.length Sys.argv == 5
    then
      let modeStr = Sys.argv.(1) in
      let mode =
        if String.equal modeStr "loss" then Loss
        else if String.equal modeStr "grad" then Grad
        else if String.equal modeStr "both" then Both
        else failwith ("Unrecognised computation mode: " ^ modeStr) in
      let g = Float.of_string Sys.argv.(2) in
      let c = match parse_string ~consume:All constr_expr (Sys.argv.(3)) with
        | Ok x -> x
        | Error msg -> failwith msg in
      let ds = match parse_string ~consume:All dims_expr (Sys.argv.(4)) with
        | Ok x -> x
        | Error msg -> failwith msg in
      (mode, g, c, ds)
    else failwith "Need four arguments: loss/grad/both flag, smoothing factor, LTL constraint and data shape"

(* Validate inputs and reconstruct tensor *)
let data =
  if Lib.Arith.equal_nata (Lib.Groups_List.prod_list Lib.Arith.monoid_mult_nat dims) (Lib.List.size_list vec)
    then Lib.Tensor.tensor_construct dims vec
    else failwith "Input not a valid tensor"

(* Compute loss/gradient ove  r the whole data *)
let t = Lib.Arith.zero_nat
let w =
  match List.nth_opt (Lib.Tensor.dims data) 0 with
  | Some d -> d
  | None -> failwith "Data has no dimensions"

(* Compute and print loss and gradients in the same format as Haskell executable *)
let () =
  if mode == Loss || mode == Both
    then let loss = Lib.LTLLoss.l_base constr data t w gamma in
      output_all_float_le (Lib.Tensor.vec loss) Out_channel.stdout
    else ()
let () =
  if mode == Grad || mode == Both
    then let grad = Lib.LTLLoss.dL_base constr data t w gamma in
      output_all_float_le (Lib.Tensor.vec grad) Out_channel.stdout
    else ()

(* For constraint parser debugging

open Printf;;
(* Constraint string printer *)
let nat_to_int x = Z.to_int (Lib.Arith.integer_of_nat x)
let comp_to_string x = match x with
  | Lib.LTLLoss.Lequal (x, y) -> sprintf "%d <= %d" (nat_to_int x) (nat_to_int y)
  | Lib.LTLLoss.Less (x, y) -> sprintf "%d < %d" (nat_to_int x) (nat_to_int y)
  | Lib.LTLLoss.Nequal (x, y) -> sprintf "%d != %d" (nat_to_int x) (nat_to_int y)
  | Lib.LTLLoss.Equal (x, y) -> sprintf "%d == %d" (nat_to_int x) (nat_to_int y)
let rec constraint_to_string x = match x with
  | Lib.LTLLoss.Comp c -> sprintf "Comp %s" (comp_to_string c)
  | Lib.LTLLoss.And (x, y) -> sprintf "And (%s) (%s)" (constraint_to_string x) (constraint_to_string y)
  | Lib.LTLLoss.Or (x, y) -> sprintf "Or (%s) (%s)" (constraint_to_string x) (constraint_to_string y)
  | Lib.LTLLoss.Next x -> sprintf "Next (%s)" (constraint_to_string x)
  | Lib.LTLLoss.Always x -> sprintf "Always (%s)" (constraint_to_string x)
  | Lib.LTLLoss.Eventually x -> sprintf "Eventually (%s)" (constraint_to_string x)
  | Lib.LTLLoss.Until (x, y) -> sprintf "Until (%s) (%s)" (constraint_to_string x) (constraint_to_string y)
  | Lib.LTLLoss.Release (x, y) -> sprintf "Release (%s) (%s)" (constraint_to_string x) (constraint_to_string y)

(* Print parsed value or fail with error message *)
let () = match result with
  | Ok v -> print_endline (constraint_to_string v)
  | Error msg -> failwith msg
*)
