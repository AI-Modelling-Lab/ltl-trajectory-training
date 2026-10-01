module Fun : sig
  val id : 'a -> 'a
  val comp : ('a -> 'b) -> ('c -> 'a) -> 'c -> 'b
end = struct

let rec id x = (fun xa -> xa) x;;

let rec comp f g = (fun x -> f (g x));;

end;; (*struct Fun*)

module HOL : sig
  type 'a equal = {equal : 'a -> 'a -> bool}
  val equal : 'a equal -> 'a -> 'a -> bool
  val eq : 'a equal -> 'a -> 'a -> bool
end = struct

type 'a equal = {equal : 'a -> 'a -> bool};;
let equal _A = _A.equal;;

let rec eq _A a b = equal _A a b;;

end;; (*struct HOL*)

module Product_Type : sig
  val apsnd : ('a -> 'b) -> 'c * 'a -> 'c * 'b
  val fst : 'a * 'b -> 'a
  val snd : 'a * 'b -> 'b
end = struct

let rec apsnd f (x, y) = (x, f y);;

let rec fst (x1, x2) = x1;;

let rec snd (x1, x2) = x2;;

end;; (*struct Product_Type*)

module Orderings : sig
  type 'a ord = {less_eq : 'a -> 'a -> bool; less : 'a -> 'a -> bool}
  val less_eq : 'a ord -> 'a -> 'a -> bool
  val less : 'a ord -> 'a -> 'a -> bool
  val max : 'a ord -> 'a -> 'a -> 'a
  val min : 'a ord -> 'a -> 'a -> 'a
end = struct

type 'a ord = {less_eq : 'a -> 'a -> bool; less : 'a -> 'a -> bool};;
let less_eq _A = _A.less_eq;;
let less _A = _A.less;;

let rec max _A a b = (if less_eq _A a b then b else a);;

let rec min _A a b = (if less_eq _A a b then a else b);;

end;; (*struct Orderings*)

module Arith : sig
  type int = Int_of_integer of Z.t
  val integer_of_int : int -> Z.t
  val plus_inta : int -> int -> int
  type 'a plus = {plus : 'a -> 'a -> 'a}
  val plus : 'a plus -> 'a -> 'a -> 'a
  val plus_int : int plus
  val minus_inta : int -> int -> int
  type 'a minus = {minus : 'a -> 'a -> 'a}
  val minus : 'a minus -> 'a -> 'a -> 'a
  val minus_int : int minus
  type 'a semigroup_add = {plus_semigroup_add : 'a plus}
  val semigroup_add_int : int semigroup_add
  type nat = Nat of Z.t
  val integer_of_nat : nat -> Z.t
  val equal_nata : nat -> nat -> bool
  val equal_nat : nat HOL.equal
  type num = One | Bit0 of num | Bit1 of num
  val one_nata : nat
  type 'a one = {one : 'a}
  val one : 'a one -> 'a
  val one_nat : nat one
  val times_nata : nat -> nat -> nat
  type 'a times = {times : 'a -> 'a -> 'a}
  val times : 'a times -> 'a -> 'a -> 'a
  type 'a power = {one_power : 'a one; times_power : 'a times}
  val times_nat : nat times
  val power_nat : nat power
  val less_eq_nat : nat -> nat -> bool
  val less_nat : nat -> nat -> bool
  val ord_nat : nat Orderings.ord
  type 'a semigroup_mult = {times_semigroup_mult : 'a times}
  type 'a monoid_mult =
    {semigroup_mult_monoid_mult : 'a semigroup_mult;
      power_monoid_mult : 'a power}
  val semigroup_mult_nat : nat semigroup_mult
  val monoid_mult_nat : nat monoid_mult
  val ord_integer : Z.t Orderings.ord
  type 'a zero = {zero : 'a}
  val zero : 'a zero -> 'a
  val plus_nat : nat -> nat -> nat
  val suc : nat -> nat
  val one_int : int
  val zero_nat : nat
  val divmod_integer : Z.t -> Z.t -> Z.t * Z.t
  val nat_of_integer : Z.t -> nat
  val minus_nat : nat -> nat -> nat
  val divide_integer : Z.t -> Z.t -> Z.t
  val divide_nat : nat -> nat -> nat
  val modulo_integer : Z.t -> Z.t -> Z.t
  val modulo_nat : nat -> nat -> nat
end = struct

type int = Int_of_integer of Z.t;;

let rec integer_of_int (Int_of_integer k) = k;;

let rec plus_inta
  k l = Int_of_integer (Z.add (integer_of_int k) (integer_of_int l));;

type 'a plus = {plus : 'a -> 'a -> 'a};;
let plus _A = _A.plus;;

let plus_int = ({plus = plus_inta} : int plus);;

let rec minus_inta
  k l = Int_of_integer (Z.sub (integer_of_int k) (integer_of_int l));;

type 'a minus = {minus : 'a -> 'a -> 'a};;
let minus _A = _A.minus;;

let minus_int = ({minus = minus_inta} : int minus);;

type 'a semigroup_add = {plus_semigroup_add : 'a plus};;

let semigroup_add_int = ({plus_semigroup_add = plus_int} : int semigroup_add);;

type nat = Nat of Z.t;;

let rec integer_of_nat (Nat x) = x;;

let rec equal_nata m n = Z.equal (integer_of_nat m) (integer_of_nat n);;

let equal_nat = ({HOL.equal = equal_nata} : nat HOL.equal);;

type num = One | Bit0 of num | Bit1 of num;;

let one_nata : nat = Nat (Z.of_int 1);;

type 'a one = {one : 'a};;
let one _A = _A.one;;

let one_nat = ({one = one_nata} : nat one);;

let rec times_nata m n = Nat (Z.mul (integer_of_nat m) (integer_of_nat n));;

type 'a times = {times : 'a -> 'a -> 'a};;
let times _A = _A.times;;

type 'a power = {one_power : 'a one; times_power : 'a times};;

let times_nat = ({times = times_nata} : nat times);;

let power_nat = ({one_power = one_nat; times_power = times_nat} : nat power);;

let rec less_eq_nat m n = Z.leq (integer_of_nat m) (integer_of_nat n);;

let rec less_nat m n = Z.lt (integer_of_nat m) (integer_of_nat n);;

let ord_nat =
  ({Orderings.less_eq = less_eq_nat; Orderings.less = less_nat} :
    nat Orderings.ord);;

type 'a semigroup_mult = {times_semigroup_mult : 'a times};;

type 'a monoid_mult =
  {semigroup_mult_monoid_mult : 'a semigroup_mult;
    power_monoid_mult : 'a power};;

let semigroup_mult_nat =
  ({times_semigroup_mult = times_nat} : nat semigroup_mult);;

let monoid_mult_nat =
  ({semigroup_mult_monoid_mult = semigroup_mult_nat;
     power_monoid_mult = power_nat}
    : nat monoid_mult);;

let ord_integer =
  ({Orderings.less_eq = Z.leq; Orderings.less = Z.lt} : Z.t Orderings.ord);;

type 'a zero = {zero : 'a};;
let zero _A = _A.zero;;

let rec plus_nat m n = Nat (Z.add (integer_of_nat m) (integer_of_nat n));;

let rec suc n = plus_nat n one_nata;;

let one_int : int = Int_of_integer (Z.of_int 1);;

let zero_nat : nat = Nat Z.zero;;

let rec divmod_integer
  k l = (if Z.equal k Z.zero then (Z.zero, Z.zero)
          else (if Z.lt Z.zero l
                 then (if Z.lt Z.zero k
                        then (fun k l -> if Z.equal Z.zero l then
                               (Z.zero, l) else Z.div_rem (Z.abs k) (Z.abs l))
                               k l
                        else (let (r, s) =
                                (fun k l -> if Z.equal Z.zero l then
                                  (Z.zero, l) else Z.div_rem (Z.abs k)
                                  (Z.abs l))
                                  k l
                                in
                               (if Z.equal s Z.zero then (Z.neg r, Z.zero)
                                 else (Z.sub (Z.neg r) (Z.of_int 1),
Z.sub l s))))
                 else (if Z.equal l Z.zero then (Z.zero, k)
                        else Product_Type.apsnd Z.neg
                               (if Z.lt k Z.zero
                                 then (fun k l -> if Z.equal Z.zero l then
(Z.zero, l) else Z.div_rem (Z.abs k) (Z.abs l))
k l
                                 else (let (r, s) =
 (fun k l -> if Z.equal Z.zero l then (Z.zero, l) else Z.div_rem (Z.abs k)
   (Z.abs l))
   k l
 in
(if Z.equal s Z.zero then (Z.neg r, Z.zero)
  else (Z.sub (Z.neg r) (Z.of_int 1), Z.sub (Z.neg l) s)))))));;

let rec nat_of_integer k = Nat (Orderings.max ord_integer Z.zero k);;

let rec minus_nat
  m n = Nat (Orderings.max ord_integer Z.zero
              (Z.sub (integer_of_nat m) (integer_of_nat n)));;

let rec divide_integer k l = Product_Type.fst (divmod_integer k l);;

let rec divide_nat
  m n = Nat (divide_integer (integer_of_nat m) (integer_of_nat n));;

let rec modulo_integer k l = Product_Type.snd (divmod_integer k l);;

let rec modulo_nat
  m n = Nat (modulo_integer (integer_of_nat m) (integer_of_nat n));;

end;; (*struct Arith*)

module Rat : sig
  type rat = Frct of (Arith.int * Arith.int)
  val of_int : Arith.int -> rat
  val quotient_of : rat -> Arith.int * Arith.int
end = struct

type rat = Frct of (Arith.int * Arith.int);;

let rec of_int a = Frct (a, Arith.one_int);;

let rec quotient_of (Frct x) = x;;

end;; (*struct Rat*)

module List : sig
  val upt : Arith.nat -> Arith.nat -> Arith.nat list
  val drop : Arith.nat -> 'a list -> 'a list
  val maps : ('a -> 'b list) -> 'a list -> 'b list
  val take : Arith.nat -> 'a list -> 'a list
  val foldr : ('a -> 'b -> 'b) -> 'a list -> 'b -> 'b
  val member : 'a HOL.equal -> 'a list -> 'a -> bool
  val map : ('a -> 'b) -> 'a list -> 'b list
  val gen_length : Arith.nat -> 'a list -> Arith.nat
  val list_all : ('a -> bool) -> 'a list -> bool
  val size_list : 'a list -> Arith.nat
  val equal_list : 'a HOL.equal -> 'a list -> 'a list -> bool
end = struct

let rec upt
  i j = (if Arith.less_nat i j then i :: upt (Arith.suc i) j else []);;

let rec drop
  n x1 = match n, x1 with n, [] -> []
    | n, x :: xs ->
        (if Arith.equal_nata n Arith.zero_nat then x :: xs
          else drop (Arith.minus_nat n Arith.one_nata) xs);;

let rec maps f x1 = match f, x1 with f, [] -> []
               | f, x :: xs -> f x @ maps f xs;;

let rec take
  n x1 = match n, x1 with n, [] -> []
    | n, x :: xs ->
        (if Arith.equal_nata n Arith.zero_nat then []
          else x :: take (Arith.minus_nat n Arith.one_nata) xs);;

let rec foldr f x1 = match f, x1 with f, [] -> Fun.id
                | f, x :: xs -> Fun.comp (f x) (foldr f xs);;

let rec member _A x0 y = match x0, y with [], y -> false
                    | x :: xs, y -> HOL.eq _A x y || member _A xs y;;

let rec map f x1 = match f, x1 with f, [] -> []
              | f, x21 :: x22 -> f x21 :: map f x22;;

let rec gen_length
  n x1 = match n, x1 with n, x :: xs -> gen_length (Arith.suc n) xs
    | n, [] -> n;;

let rec list_all p x1 = match p, x1 with p, [] -> true
                   | p, x :: xs -> p x && list_all p xs;;

let rec size_list x = gen_length Arith.zero_nat x;;

let rec equal_list _A
  x0 x1 = match x0, x1 with [], x21 :: x22 -> false
    | x21 :: x22, [] -> false
    | x21 :: x22, y21 :: y22 -> HOL.eq _A x21 y21 && equal_list _A x22 y22
    | [], [] -> true;;

end;; (*struct List*)

module Code_Real_Approx_By_Float : sig
  val real_of_int : Arith.int -> float
end = struct

let rec real_of_int
  x = Fun.comp (fun a -> Stdlib.float_of_int (Z.to_int a)) Arith.integer_of_int
        x;;

end;; (*struct Code_Real_Approx_By_Float*)

module Real : sig
  val one_real : float Arith.one
  val plus_real : float Arith.plus
  val zero_real : float Arith.zero
  val minus_real : float Arith.minus
  val times_real : float Arith.times
  val ord_real : float Orderings.ord
  val semigroup_add_real : float Arith.semigroup_add
  val ratreal : Rat.rat -> float
  val sgn_real : float -> float
  val inverse_real : float -> float
end = struct

let one_real = ({Arith.one = 1.0} : float Arith.one);;

let plus_real = ({Arith.plus = Stdlib.( +. )} : float Arith.plus);;

let zero_real = ({Arith.zero = 0.0} : float Arith.zero);;

let minus_real = ({Arith.minus = Stdlib.( -. )} : float Arith.minus);;

let times_real = ({Arith.times = Stdlib.( *. )} : float Arith.times);;

let ord_real =
  ({Orderings.less_eq = Stdlib.(<=); Orderings.less = Stdlib.(<)} :
    float Orderings.ord);;

let semigroup_add_real =
  ({Arith.plus_semigroup_add = plus_real} : float Arith.semigroup_add);;

let rec ratreal
  r = (let (p, q) = Rat.quotient_of r in
        Stdlib.( /. ) (Code_Real_Approx_By_Float.real_of_int p)
          (Code_Real_Approx_By_Float.real_of_int q));;

let rec sgn_real
  a = (if Stdlib.(=) a 0.0 then 0.0
        else (if Stdlib.(<) 0.0 a then 1.0 else Stdlib.( ~-. ) 1.0));;

let rec inverse_real r = Stdlib.( /. ) 1.0 r;;

end;; (*struct Real*)

module IArray : sig
  val sub : 'a array -> Arith.nat -> 'a
  val length : 'a array -> Arith.nat
  val list_of : 'a array -> 'a list
  val map_iarray : ('a -> 'b) -> 'a array -> 'b array
end = struct

let rec sub asa n = Array.get asa (Z.to_int (Arith.integer_of_nat n));;

let rec length asa = Arith.nat_of_integer (Z.of_int (Array.length asa));;

let rec list_of
  asa = List.map (sub asa) (List.upt Arith.zero_nat (length asa));;

let rec map_iarray f asa = Array.of_list (List.map f (list_of asa));;

end;; (*struct IArray*)

module Option : sig
  val the : 'a option -> 'a
  val map_option : ('a -> 'b) -> 'a option -> 'b option
end = struct

let rec the (Some x2) = x2;;

let rec map_option f x1 = match f, x1 with f, None -> None
                     | f, Some x2 -> Some (f x2);;

end;; (*struct Option*)

module IArraySupplement : sig
  val genArray : Arith.nat -> (Arith.nat -> 'a) -> 'a array
  val map2 : ('a -> 'b -> 'c) -> 'a array -> 'b array -> 'c array
  val concat : ('a array) list -> 'a array
  val subarray : Arith.nat -> Arith.nat -> 'a array -> 'a array
  val replicate : Arith.nat -> 'a -> 'a array
end = struct

let rec genArray
  l f = Array.init (Z.to_int (Arith.integer_of_nat
                                l)) (fun i -> (Fun.comp f
         Arith.nat_of_integer) (Z.of_int i));;

let rec map2
  f x y =
    genArray (Orderings.min Arith.ord_nat (IArray.length x) (IArray.length y))
      (fun i -> f (IArray.sub x i) (IArray.sub y i));;

let rec concat xs = Array.of_list (List.maps IArray.list_of xs);;

let rec subarray
  i n x =
    Array.sub x (Z.to_int (Arith.integer_of_nat
                             i)) (Z.to_int (Arith.integer_of_nat n));;

let rec replicate i x = genArray i (fun _ -> x);;

end;; (*struct IArraySupplement*)

module Groups_List : sig
  val prod_list : 'a Arith.monoid_mult -> 'a list -> 'a
end = struct

let rec prod_list _A
  xs = List.foldr (Arith.times _A.Arith.power_monoid_mult.Arith.times_power) xs
         (Arith.one _A.Arith.power_monoid_mult.Arith.one_power);;

end;; (*struct Groups_List*)

module TensorIArray : sig
  type 'a tensor = Abs_tensor of (Arith.nat list * 'a array)
  val rep_tensor : 'a tensor -> Arith.nat list * 'a array
  val vec : 'a tensor -> 'a array
  val dims : 'a tensor -> Arith.nat list
  val compute_index : Arith.nat list -> Arith.nat list -> Arith.nat
  val valid_index : Arith.nat list -> Arith.nat list -> bool
  val lookup : 'a tensor -> Arith.nat list -> 'a
  val safe_Tensor : Arith.nat list * 'a array -> 'a tensor
  val tensor_unop : ('b -> 'a) -> 'b tensor -> 'a tensor
  val tensor_binop : ('b -> 'c -> 'a) -> 'b tensor -> 'c tensor -> 'a tensor
  val decode_index : Arith.nat -> Arith.nat list -> Arith.nat list
  val tensor_construct : Arith.nat list -> 'a list -> 'a tensor
  val tensor_from_list : 'a list -> 'a tensor
  val tensor_replicate : Arith.nat list -> 'a -> 'a tensor
  val subtensor_combine : Arith.nat list -> 'a tensor list -> 'a tensor option
  val tensor_from_lookup : Arith.nat list -> (Arith.nat list -> 'a) -> 'a tensor
  val fixed_length_subarray : 'a array -> Arith.nat -> Arith.nat -> 'a array
end = struct

type 'a tensor = Abs_tensor of (Arith.nat list * 'a array);;

let rec rep_tensor (Abs_tensor x) = x;;

let rec vec t = Product_Type.snd (rep_tensor t);;

let rec dims t = Product_Type.fst (rep_tensor t);;

let rec compute_index
  x0 x1 = match x0, x1 with
    i :: is, d :: ds ->
      Arith.plus_nat
        (Arith.times_nata i (Groups_List.prod_list Arith.monoid_mult_nat ds))
        (compute_index is ds)
    | i :: is, [] -> failwith "undefined (compute_index 2nd Nil)"
    | [], d :: ds -> failwith "undefined (compute_index 1st Nil)"
    | [], [] -> Arith.zero_nat;;

let rec valid_index
  x0 x1 = match x0, x1 with
    i :: is, d :: ds -> Arith.less_nat i d && valid_index is ds
    | i :: is, [] -> false
    | [], d :: ds -> false
    | [], [] -> true;;

let rec lookup
  t is =
    (if valid_index is (dims t)
      then IArray.sub (vec t) (compute_index is (dims t))
      else failwith "undefined (lookup)");;

let rec safe_Tensor
  x = Abs_tensor
        (if Arith.equal_nata (IArray.length (Product_Type.snd x))
              (Groups_List.prod_list Arith.monoid_mult_nat (Product_Type.fst x))
          then x else rep_tensor (failwith "undefined (safe_Tensor else case)"));;

let rec tensor_unop f x = Abs_tensor (dims x, IArray.map_iarray f (vec x));;

let rec tensor_binop
  f a b =
    Abs_tensor
      (if List.equal_list Arith.equal_nat (dims a) (dims b)
        then (dims a, IArraySupplement.map2 f (vec a) (vec b))
        else rep_tensor (failwith "undefined (tensor_binop else case)"));;

let rec decode_index
  i x1 = match i, x1 with
    i, d :: ds ->
      (if Arith.equal_nata i Arith.zero_nat
        then Arith.zero_nat :: decode_index Arith.zero_nat ds
        else Arith.divide_nat (Arith.suc (Arith.minus_nat i Arith.one_nata))
               (Groups_List.prod_list Arith.monoid_mult_nat ds) ::
               decode_index
                 (Arith.modulo_nat
                   (Arith.suc (Arith.minus_nat i Arith.one_nata))
                   (Groups_List.prod_list Arith.monoid_mult_nat ds))
                 ds)
    | i, [] ->
        (if Arith.equal_nata i Arith.zero_nat then []
          else failwith "undefined (decode_index 2nd Nil)");;

let rec tensor_construct
  ds es =
    Abs_tensor
      (if Arith.equal_nata (Groups_List.prod_list Arith.monoid_mult_nat ds)
            (List.size_list es)
        then (ds, Array.of_list es) else rep_tensor (failwith "undefined (tensor_construct else case)"));;

let rec tensor_from_list
  xs = Abs_tensor ([List.size_list xs], Array.of_list xs);;

let rec tensor_replicate
  ds a =
    Abs_tensor
      (ds, IArraySupplement.replicate
             (Groups_List.prod_list Arith.monoid_mult_nat ds) a);;

let rec subtensor_combine
  ds asa =
    (if List.list_all (fun a -> List.equal_list Arith.equal_nat (dims a) ds) asa
      then Some (safe_Tensor
                  (List.size_list asa :: ds,
                    IArraySupplement.concat (List.map vec asa)))
      else None);;

let rec tensor_from_lookup
  ds f =
    Abs_tensor
      (ds, IArraySupplement.genArray
             (Groups_List.prod_list Arith.monoid_mult_nat ds)
             (Fun.comp f (fun i -> decode_index i ds)));;

let rec fixed_length_subarray
  xs l i = IArraySupplement.subarray (Arith.times_nata l i) l xs;;

end;; (*struct TensorIArray*)

module Tensor : sig
  type 'a tensor = Tensor_list of 'a TensorIArray.tensor
  val tensor_iarray : 'a tensor -> 'a TensorIArray.tensor
  val vec : 'a tensor -> 'a list
  val dims : 'a tensor -> Arith.nat list
  val lookup : 'a tensor -> Arith.nat list -> 'a
  val valid_index : Arith.nat list -> Arith.nat list -> bool
  val iarray : 'a tensor -> 'a array
  val tensor_replicate : Arith.nat list -> 'a -> 'a tensor
  val tensor0 : 'a Arith.zero -> Arith.nat list -> 'a tensor
  val tensor1 : 'a Arith.one -> Arith.nat list -> 'a tensor
  val tensor_from_lookup : Arith.nat list -> (Arith.nat list -> 'a) -> 'a tensor
  val tensor_unop : ('b -> 'a) -> 'b tensor -> 'a tensor
  val safe_Tensor_list : 'a TensorIArray.tensor -> 'a tensor
  val tensor_binop : ('b -> 'c -> 'a) -> 'b tensor -> 'c tensor -> 'a tensor
  val compute_index : Arith.nat list -> Arith.nat list -> Arith.nat
  val tensor_hadamard : 'a Arith.times -> 'a tensor -> 'a tensor -> 'a tensor
  val repeat_subtensor : 'a tensor -> Arith.nat list -> 'a tensor
  val tensor_construct : Arith.nat list -> 'a list -> 'a tensor
  val tensor_from_list : 'a list -> 'a tensor
  val subtensor_combine : Arith.nat list -> 'a tensor list -> 'a tensor option
  val safe_iterated_subtensor : Arith.nat list -> 'a tensor -> 'a tensor option
  val plus_tensor :
    'a Arith.semigroup_add -> 'a tensor -> 'a tensor -> 'a tensor
  val iterated_subtensor_backward :
    'a Arith.one * 'a Arith.zero -> Arith.nat list -> 'a tensor -> 'a tensor
  val minus_tensor : 'a Arith.minus -> 'a tensor -> 'a tensor -> 'a tensor
end = struct

type 'a tensor = Tensor_list of 'a TensorIArray.tensor;;

let rec tensor_iarray (Tensor_list x) = x;;

let rec vec t = IArray.list_of (TensorIArray.vec (tensor_iarray t));;

let rec dims t = TensorIArray.dims (tensor_iarray t);;

let rec lookup t is = TensorIArray.lookup (tensor_iarray t) is;;

let rec valid_index
  x0 x1 = match x0, x1 with
    i :: is, d :: ds -> Arith.less_nat i d && valid_index is ds
    | i :: is, [] -> false
    | [], d :: ds -> false
    | [], [] -> true;;

let rec iarray t = TensorIArray.vec (tensor_iarray t);;

let rec tensor_replicate
  ds a = Tensor_list (TensorIArray.tensor_replicate ds a);;

let rec tensor0 _A d = tensor_replicate d (Arith.zero _A);;

let rec tensor1 _A ds = tensor_replicate ds (Arith.one _A);;

let rec tensor_from_lookup
  ds f = Tensor_list (TensorIArray.tensor_from_lookup ds f);;

let rec tensor_unop
  f x = Tensor_list (TensorIArray.tensor_unop f (tensor_iarray x));;

let rec safe_Tensor_list x = Tensor_list x;;

let rec tensor_binop
  f x y =
    Tensor_list
      (if List.equal_list Arith.equal_nat (dims x) (dims y)
        then TensorIArray.tensor_binop f (tensor_iarray x) (tensor_iarray y)
        else tensor_iarray (failwith "undefined (tensor_binop(L) else case)"));;

let rec compute_index
  x0 x1 = match x0, x1 with [], [] -> Arith.zero_nat
    | [], d :: ds -> failwith "undefined (compute_index(L) 1st Nil)"
    | i :: is, [] -> failwith "undefined (compute_index(L) 2nd Nil)"
    | i :: is, d :: ds ->
        Arith.plus_nat
          (Arith.times_nata i (Groups_List.prod_list Arith.monoid_mult_nat ds))
          (compute_index is ds);;

let rec tensor_hadamard _A a b = tensor_binop (Arith.times _A) a b;;

let rec repeat_subtensor
  a ds =
    tensor_from_lookup (ds @ dims a)
      (fun is -> lookup a (List.drop (List.size_list ds) is));;

let rec tensor_construct
  ds es =
    Tensor_list
      (if Arith.equal_nata (Groups_List.prod_list Arith.monoid_mult_nat ds)
            (List.size_list es)
        then TensorIArray.tensor_construct ds es
        else tensor_iarray (failwith "undefined (tensor_construct(L) else case)"));;

let rec tensor_from_list xs = Tensor_list (TensorIArray.tensor_from_list xs);;

let rec subtensor_combine
  ds asa =
    Option.map_option safe_Tensor_list
      (TensorIArray.subtensor_combine ds (List.map tensor_iarray asa));;

let rec safe_iterated_subtensor
  is t =
    (if valid_index is (List.take (List.size_list is) (dims t))
      then (if List.member Arith.equal_nat
                 (List.drop (List.size_list is) (dims t)) Arith.zero_nat
             then Some (safe_Tensor_list
                         (TensorIArray.safe_Tensor
                           (List.drop (List.size_list is) (dims t),
                             Array.of_list [])))
             else Some (safe_Tensor_list
                         (TensorIArray.safe_Tensor
                           (List.drop (List.size_list is) (dims t),
                             TensorIArray.fixed_length_subarray (iarray t)
                               (Groups_List.prod_list Arith.monoid_mult_nat
                                 (List.drop (List.size_list is) (dims t)))
                               (compute_index is
                                 (List.take (List.size_list is) (dims t)))))))
      else None);;

let rec plus_tensor _A
  x y = tensor_binop (Arith.plus _A.Arith.plus_semigroup_add) x y;;

let rec iterated_subtensor_backward (_A1, _A2)
  is a =
    tensor_from_lookup (dims a)
      (fun js ->
        (if List.equal_list Arith.equal_nat (List.take (List.size_list is) js)
              is
          then Arith.one _A1 else Arith.zero _A2));;

let rec minus_tensor _A a b = tensor_binop (Arith.minus _A) a b;;

end;; (*struct Tensor*)

module SmoothComp : sig
  val max_gamma : float -> float -> float -> float
  val min_gamma : float -> float -> float -> float
  val bell_gamma : float -> float -> float
  val rect_gamma : float -> float -> float
  val dMax_first : float -> float -> float
  val dMin_first : float -> float -> float
  val dBell_gamma : float -> float -> float
  val dMax_second : float -> float -> float
  val dMin_second : float -> float -> float
  val dMax_gamma_second : float -> float -> float -> float
  val dRect_gamma : float -> float -> float
  val lequal_gamma : float -> float -> float -> float
  val nequal_gamma : float -> float -> float -> float
  val dMax_gamma_first : float -> float -> float -> float
  val dMin_gamma_first : float -> float -> float -> float
  val dMin_gamma_second : float -> float -> float -> float
end = struct

let rec max_gamma
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then Orderings.max Real.ord_real a b
      else Stdlib.( *. ) gamma
             (if Stdlib.(<) a b
               then Stdlib.( +. ) (Stdlib.( /. ) a gamma)
                      (Stdlib.log
                        (Stdlib.( +. ) 1.0
                          (Stdlib.exp
                            (Stdlib.( /. ) (Stdlib.( -. ) b a) gamma))))
               else (if Stdlib.(<) b a
                      then Stdlib.( +. ) (Stdlib.( /. ) b gamma)
                             (Stdlib.log
                               (Stdlib.( +. ) 1.0
                                 (Stdlib.exp
                                   (Stdlib.( /. ) (Stdlib.( -. ) a b) gamma))))
                      else Stdlib.( +. ) (Stdlib.( /. ) a gamma)
                             (Stdlib.log
                               (Real.ratreal
                                 (Rat.of_int
                                   (Arith.Int_of_integer (Z.of_int 2))))))));;

let rec min_gamma
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then Orderings.min Real.ord_real a b
      else Stdlib.( *. ) (Stdlib.( ~-. ) gamma)
             (if Stdlib.(<) b a
               then Stdlib.( +. ) (Stdlib.( /. ) (Stdlib.( ~-. ) a) gamma)
                      (Stdlib.log
                        (Stdlib.( +. ) 1.0
                          (Stdlib.exp
                            (Stdlib.( /. ) (Stdlib.( -. ) a b) gamma))))
               else (if Stdlib.(<) a b
                      then Stdlib.( +. )
                             (Stdlib.( /. ) (Stdlib.( ~-. ) b) gamma)
                             (Stdlib.log
                               (Stdlib.( +. ) 1.0
                                 (Stdlib.exp
                                   (Stdlib.( /. ) (Stdlib.( -. ) b a) gamma))))
                      else Stdlib.( +. )
                             (Stdlib.( /. ) (Stdlib.( ~-. ) a) gamma)
                             (Stdlib.log
                               (Real.ratreal
                                 (Rat.of_int
                                   (Arith.Int_of_integer (Z.of_int 2))))))));;

let rec bell_gamma
  gamma x =
    (if Stdlib.(<=) gamma 0.0 then (if Stdlib.(=) x 0.0 then 1.0 else 0.0)
      else Stdlib.( /. ) 1.0
             (Stdlib.exp
               (Stdlib.( /. ) (Stdlib.( *. ) x x)
                 (Stdlib.( *. )
                   (Real.ratreal
                     (Rat.of_int (Arith.Int_of_integer (Z.of_int 2))))
                   gamma))));;

let rec rect_gamma gamma x = max_gamma gamma 0.0 x;;

let rec dMax_first
  a b = Stdlib.( *. )
          (Stdlib.( /. ) 1.0
            (Real.ratreal (Rat.of_int (Arith.Int_of_integer (Z.of_int 2)))))
          (Stdlib.( +. ) 1.0
            (if Stdlib.(=) a b then 0.0
              else Real.sgn_real (Stdlib.( -. ) a b)));;

let rec dMin_first
  a b = Stdlib.( *. )
          (Stdlib.( /. ) 1.0
            (Real.ratreal (Rat.of_int (Arith.Int_of_integer (Z.of_int 2)))))
          (Stdlib.( -. ) 1.0
            (if Stdlib.(=) a b then 0.0
              else Real.sgn_real (Stdlib.( -. ) a b)));;

let rec dBell_gamma
  gamma x =
    Stdlib.( /. ) (Stdlib.( ~-. ) x)
      (Stdlib.( *. ) gamma
        (Stdlib.exp
          (Stdlib.( /. ) (Stdlib.( *. ) x x)
            (Stdlib.( *. )
              (Real.ratreal (Rat.of_int (Arith.Int_of_integer (Z.of_int 2))))
              gamma))));;

let rec dMax_second
  a b = Stdlib.( *. )
          (Stdlib.( /. ) 1.0
            (Real.ratreal (Rat.of_int (Arith.Int_of_integer (Z.of_int 2)))))
          (Stdlib.( +. ) 1.0
            (if Stdlib.(=) a b then 0.0
              else Real.sgn_real (Stdlib.( -. ) b a)));;

let rec dMin_second
  a b = Stdlib.( *. )
          (Stdlib.( /. ) 1.0
            (Real.ratreal (Rat.of_int (Arith.Int_of_integer (Z.of_int 2)))))
          (Stdlib.( -. ) 1.0
            (if Stdlib.(=) a b then 0.0
              else Real.sgn_real (Stdlib.( -. ) b a)));;

let rec dMax_gamma_second
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then dMax_second a b
      else Real.inverse_real
             (Stdlib.( +. ) 1.0
               (Stdlib.exp (Stdlib.( /. ) (Stdlib.( -. ) a b) gamma))));;

let rec dRect_gamma
  gamma x =
    (if Stdlib.(<=) gamma 0.0 then 1.0 else dMax_gamma_second gamma 0.0 x);;

let rec lequal_gamma gamma a b = rect_gamma gamma (Stdlib.( -. ) a b);;

let rec nequal_gamma gamma a b = bell_gamma gamma (Stdlib.( -. ) a b);;

let rec dMax_gamma_first
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then dMax_first a b
      else Real.inverse_real
             (Stdlib.( +. ) 1.0
               (Stdlib.exp (Stdlib.( /. ) (Stdlib.( -. ) b a) gamma))));;

let rec dMin_gamma_first
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then dMin_first a b
      else Real.inverse_real
             (Stdlib.( +. ) 1.0
               (Stdlib.exp (Stdlib.( /. ) (Stdlib.( -. ) a b) gamma))));;

let rec dMin_gamma_second
  gamma a b =
    (if Stdlib.(<=) gamma 0.0 then dMin_second a b
      else Real.inverse_real
             (Stdlib.( +. ) 1.0
               (Stdlib.exp (Stdlib.( /. ) (Stdlib.( -. ) b a) gamma))));;

end;; (*struct SmoothComp*)

module SmoothCompTensor : sig
  val tensor_Max_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_Min_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_Nequal_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_Lequal_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_Less_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_Equal_gamma :
    float -> float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dMax_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dMin_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dNequal_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dLequal_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dLess_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
  val tensor_dEqual_gamma_chain :
    float ->
      float Tensor.tensor ->
        float Tensor.tensor ->
          float Tensor.tensor -> float Tensor.tensor -> float Tensor.tensor
end = struct

let rec tensor_Max_gamma
  gamma a b = Tensor.tensor_binop (SmoothComp.max_gamma gamma) a b;;

let rec tensor_Min_gamma
  gamma a b = Tensor.tensor_binop (SmoothComp.min_gamma gamma) a b;;

let rec tensor_Nequal_gamma
  gamma a b = Tensor.tensor_binop (SmoothComp.nequal_gamma gamma) a b;;

let rec tensor_Lequal_gamma
  gamma a b = Tensor.tensor_binop (SmoothComp.lequal_gamma gamma) a b;;

let rec tensor_Less_gamma
  gamma a b =
    tensor_Max_gamma gamma (tensor_Lequal_gamma gamma a b)
      (tensor_Nequal_gamma gamma a b);;

let rec tensor_Equal_gamma
  gamma a b =
    tensor_Max_gamma gamma (tensor_Lequal_gamma gamma a b)
      (tensor_Lequal_gamma gamma b a);;

let rec tensor_dMax_gamma_chain
  gamma a dA b dB =
    Tensor.plus_tensor Real.semigroup_add_real
      (Tensor.tensor_hadamard Real.times_real
        (Tensor.repeat_subtensor
          (Tensor.tensor_binop (SmoothComp.dMax_gamma_first gamma) a b)
          (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dA)))
        dA)
      (Tensor.tensor_hadamard Real.times_real
        (Tensor.repeat_subtensor
          (Tensor.tensor_binop (SmoothComp.dMax_gamma_second gamma) a b)
          (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dB)))
        dB);;

let rec tensor_dMin_gamma_chain
  gamma a dA b dB =
    Tensor.plus_tensor Real.semigroup_add_real
      (Tensor.tensor_hadamard Real.times_real
        (Tensor.repeat_subtensor
          (Tensor.tensor_binop (SmoothComp.dMin_gamma_first gamma) a b)
          (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dA)))
        dA)
      (Tensor.tensor_hadamard Real.times_real
        (Tensor.repeat_subtensor
          (Tensor.tensor_binop (SmoothComp.dMin_gamma_second gamma) a b)
          (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dB)))
        dB);;

let rec tensor_dNequal_gamma_chain
  gamma a dA b dB =
    Tensor.tensor_hadamard Real.times_real
      (Tensor.repeat_subtensor
        (Tensor.tensor_unop (SmoothComp.dBell_gamma gamma)
          (Tensor.minus_tensor Real.minus_real a b))
        (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dA)))
      (Tensor.minus_tensor Real.minus_real dA dB);;

let rec tensor_dLequal_gamma_chain
  gamma a dA b dB =
    Tensor.tensor_hadamard Real.times_real
      (Tensor.repeat_subtensor
        (Tensor.tensor_unop (SmoothComp.dRect_gamma gamma)
          (Tensor.minus_tensor Real.minus_real a b))
        (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims dA)))
      (Tensor.minus_tensor Real.minus_real dA dB);;

let rec tensor_dLess_gamma_chain
  gamma a dA b dB =
    tensor_dMax_gamma_chain gamma (tensor_Lequal_gamma gamma a b)
      (tensor_dLequal_gamma_chain gamma a dA b dB)
      (tensor_Nequal_gamma gamma a b)
      (tensor_dNequal_gamma_chain gamma a dA b dB);;

let rec tensor_dEqual_gamma_chain
  gamma a dA b dB =
    tensor_dMax_gamma_chain gamma (tensor_Lequal_gamma gamma a b)
      (tensor_dLequal_gamma_chain gamma a dA b dB)
      (tensor_Lequal_gamma gamma b a)
      (tensor_dLequal_gamma_chain gamma b dB a dA);;

end;; (*struct SmoothCompTensor*)

module LTLLoss : sig
  type comp = Less of Arith.nat * Arith.nat | Lequal of Arith.nat * Arith.nat |
    Equal of Arith.nat * Arith.nat | Nequal of Arith.nat * Arith.nat
  type constrainta = Comp of comp | And of constrainta * constrainta |
    Or of constrainta * constrainta | Next of constrainta |
    WeakNext of constrainta | Always of constrainta | Eventually of constrainta
    | Until of constrainta * constrainta | Release of constrainta * constrainta
  val comp_inverse : comp -> comp
  val nota : constrainta -> constrainta
  val max_state_idx_comp : comp -> Arith.nat
  val max_state_idx_constraint : constrainta -> Arith.nat
  val l_base_constraint_unsafe :
    constrainta ->
      float Tensor.tensor ->
        Arith.nat -> Arith.nat -> float -> float Tensor.tensor
  val l_base :
    constrainta ->
      float Tensor.tensor ->
        Arith.nat -> Arith.nat -> float -> float Tensor.tensor
  val dL_base_constraint_unsafe :
    constrainta ->
      float Tensor.tensor ->
        Arith.nat -> Arith.nat -> float -> float Tensor.tensor
  val dL_base :
    constrainta ->
      float Tensor.tensor ->
        Arith.nat -> Arith.nat -> float -> float Tensor.tensor
end = struct

type comp = Less of Arith.nat * Arith.nat | Lequal of Arith.nat * Arith.nat |
  Equal of Arith.nat * Arith.nat | Nequal of Arith.nat * Arith.nat;;

type constrainta = Comp of comp | And of constrainta * constrainta |
  Or of constrainta * constrainta | Next of constrainta |
  WeakNext of constrainta | Always of constrainta | Eventually of constrainta |
  Until of constrainta * constrainta | Release of constrainta * constrainta;;

let rec comp_inverse = function Less (x, y) -> Lequal (y, x)
                       | Lequal (x, y) -> Less (y, x)
                       | Equal (x, y) -> Nequal (x, y)
                       | Nequal (x, y) -> Equal (x, y);;

let rec nota = function Comp c -> Comp (comp_inverse c)
               | And (c1, c2) -> Or (nota c1, nota c2)
               | Or (c1, c2) -> And (nota c1, nota c2)
               | Next c -> WeakNext (nota c)
               | WeakNext c -> Next (nota c)
               | Always c -> Eventually (nota c)
               | Eventually c -> Always (nota c)
               | Until (c1, c2) -> Release (nota c1, nota c2)
               | Release (c1, c2) -> Until (nota c1, nota c2);;

let rec max_state_idx_comp
  = function Less (x, y) -> Orderings.max Arith.ord_nat y x
    | Lequal (x, y) -> Orderings.max Arith.ord_nat y x
    | Equal (x, y) -> Orderings.max Arith.ord_nat x y
    | Nequal (x, y) -> Orderings.max Arith.ord_nat x y;;

let rec max_state_idx_constraint
  = function Comp c -> max_state_idx_comp c
    | And (c1, c2) ->
        Orderings.max Arith.ord_nat (max_state_idx_constraint c1)
          (max_state_idx_constraint c2)
    | Or (c1, c2) ->
        Orderings.max Arith.ord_nat (max_state_idx_constraint c1)
          (max_state_idx_constraint c2)
    | Next c -> max_state_idx_constraint c
    | WeakNext c -> max_state_idx_constraint c
    | Always c -> max_state_idx_constraint c
    | Eventually c -> max_state_idx_constraint c
    | Until (c1, c2) ->
        Orderings.max Arith.ord_nat (max_state_idx_constraint c1)
          (max_state_idx_constraint c2)
    | Release (c1, c2) ->
        Orderings.max Arith.ord_nat (max_state_idx_constraint c1)
          (max_state_idx_constraint c2);;

let rec l_base_constraint_unsafe
  x0 a t w gamma = match x0, a, t, w, gamma with
    Comp (Lequal (v1, v2)), a, t, w, gamma ->
      SmoothCompTensor.tensor_Lequal_gamma gamma
        (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
        (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
    | Comp (Nequal (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_Nequal_gamma gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
    | Comp (Less (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_Less_gamma gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
    | Comp (Equal (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_Equal_gamma gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
    | And (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_Max_gamma gamma
          (l_base_constraint_unsafe c1 a t w gamma)
          (l_base_constraint_unsafe c2 a t w gamma)
    | Or (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_Min_gamma gamma
          (l_base_constraint_unsafe c1 a t w gamma)
          (l_base_constraint_unsafe c2 a t w gamma)
    | Next c, a, t, w, gamma ->
        (if Arith.less_eq_nat w Arith.one_nata
          then Tensor.tensor1 Real.one_real
                 (List.drop (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims a))
          else l_base_constraint_unsafe c a (Arith.suc t)
                 (Arith.minus_nat w Arith.one_nata) gamma)
    | WeakNext c, a, t, w, gamma ->
        (if Arith.less_eq_nat w Arith.one_nata
          then Tensor.tensor0 Real.zero_real
                 (List.drop (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims a))
          else l_base_constraint_unsafe c a (Arith.suc t)
                 (Arith.minus_nat w Arith.one_nata) gamma)
    | Always c, a, t, w, gamma ->
        SmoothCompTensor.tensor_Max_gamma gamma
          (l_base_constraint_unsafe c a t w gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor0 Real.zero_real
                   (List.drop (Arith.nat_of_integer (Z.of_int 2))
                     (Tensor.dims a))
            else l_base_constraint_unsafe (Always c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
    | Eventually c, a, t, w, gamma ->
        SmoothCompTensor.tensor_Min_gamma gamma
          (l_base_constraint_unsafe c a t w gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor1 Real.one_real
                   (List.drop (Arith.nat_of_integer (Z.of_int 2))
                     (Tensor.dims a))
            else l_base_constraint_unsafe (Eventually c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
    | Until (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_Min_gamma gamma
          (l_base_constraint_unsafe c2 a t w gamma)
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor0 Real.zero_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Until (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma))
    | Release (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_Min_gamma gamma
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (l_base_constraint_unsafe c2 a t w gamma))
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c2 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor1 Real.one_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Release (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma));;

let rec l_base
  c a t w gamma =
    (if Arith.equal_nata w Arith.zero_nat
      then Tensor.tensor1 Real.one_real
             (List.drop (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims a))
      else (if Tensor.valid_index
                 [Arith.minus_nat (Arith.plus_nat t w) Arith.one_nata;
                   max_state_idx_constraint c]
                 (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims a))
             then l_base_constraint_unsafe c a t w gamma
             else failwith "undefined (l_base invalid index)"));;

let rec dL_base_constraint_unsafe
  x0 a t w gamma = match x0, a, t, w, gamma with
    Comp (Lequal (v1, v2)), a, t, w, gamma ->
      SmoothCompTensor.tensor_dLequal_gamma_chain gamma
        (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
        (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
          [t; v1] a)
        (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
        (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
          [t; v2] a)
    | Comp (Nequal (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_dNequal_gamma_chain gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v1] a)
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v2] a)
    | Comp (Less (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_dLess_gamma_chain gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v1] a)
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v2] a)
    | Comp (Equal (v1, v2)), a, t, w, gamma ->
        SmoothCompTensor.tensor_dEqual_gamma_chain gamma
          (Option.the (Tensor.safe_iterated_subtensor [t; v1] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v1] a)
          (Option.the (Tensor.safe_iterated_subtensor [t; v2] a))
          (Tensor.iterated_subtensor_backward (Real.one_real, Real.zero_real)
            [t; v2] a)
    | And (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_dMax_gamma_chain gamma
          (l_base_constraint_unsafe c1 a t w gamma)
          (dL_base_constraint_unsafe c1 a t w gamma)
          (l_base_constraint_unsafe c2 a t w gamma)
          (dL_base_constraint_unsafe c2 a t w gamma)
    | Or (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_dMin_gamma_chain gamma
          (l_base_constraint_unsafe c1 a t w gamma)
          (dL_base_constraint_unsafe c1 a t w gamma)
          (l_base_constraint_unsafe c2 a t w gamma)
          (dL_base_constraint_unsafe c2 a t w gamma)
    | Next c, a, t, w, gamma ->
        (if Arith.less_eq_nat w Arith.one_nata
          then Tensor.tensor0 Real.zero_real (Tensor.dims a)
          else dL_base_constraint_unsafe c a (Arith.suc t)
                 (Arith.minus_nat w Arith.one_nata) gamma)
    | WeakNext c, a, t, w, gamma ->
        (if Arith.less_eq_nat w Arith.one_nata
          then Tensor.tensor0 Real.zero_real (Tensor.dims a)
          else dL_base_constraint_unsafe c a (Arith.suc t)
                 (Arith.minus_nat w Arith.one_nata) gamma)
    | Always c, a, t, w, gamma ->
        SmoothCompTensor.tensor_dMax_gamma_chain gamma
          (l_base_constraint_unsafe c a t w gamma)
          (dL_base_constraint_unsafe c a t w gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor0 Real.zero_real
                   (List.drop (Arith.nat_of_integer (Z.of_int 2))
                     (Tensor.dims a))
            else l_base_constraint_unsafe (Always c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor0 Real.zero_real (Tensor.dims a)
            else dL_base_constraint_unsafe (Always c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
    | Eventually c, a, t, w, gamma ->
        SmoothCompTensor.tensor_dMin_gamma_chain gamma
          (l_base_constraint_unsafe c a t w gamma)
          (dL_base_constraint_unsafe c a t w gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor1 Real.one_real
                   (List.drop (Arith.nat_of_integer (Z.of_int 2))
                     (Tensor.dims a))
            else l_base_constraint_unsafe (Eventually c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
          (if Arith.less_eq_nat w Arith.one_nata
            then Tensor.tensor0 Real.zero_real (Tensor.dims a)
            else dL_base_constraint_unsafe (Eventually c) a (Arith.suc t)
                   (Arith.minus_nat w Arith.one_nata) gamma)
    | Until (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_dMin_gamma_chain gamma
          (l_base_constraint_unsafe c2 a t w gamma)
          (dL_base_constraint_unsafe c2 a t w gamma)
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor0 Real.zero_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Until (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma))
          (SmoothCompTensor.tensor_dMax_gamma_chain gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (dL_base_constraint_unsafe c1 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor0 Real.zero_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Until (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor0 Real.zero_real (Tensor.dims a)
              else dL_base_constraint_unsafe (Until (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma))
    | Release (c1, c2), a, t, w, gamma ->
        SmoothCompTensor.tensor_dMin_gamma_chain gamma
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (l_base_constraint_unsafe c2 a t w gamma))
          (SmoothCompTensor.tensor_dMax_gamma_chain gamma
            (l_base_constraint_unsafe c1 a t w gamma)
            (dL_base_constraint_unsafe c1 a t w gamma)
            (l_base_constraint_unsafe c2 a t w gamma)
            (dL_base_constraint_unsafe c2 a t w gamma))
          (SmoothCompTensor.tensor_Max_gamma gamma
            (l_base_constraint_unsafe c2 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor1 Real.one_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Release (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma))
          (SmoothCompTensor.tensor_dMax_gamma_chain gamma
            (l_base_constraint_unsafe c2 a t w gamma)
            (dL_base_constraint_unsafe c2 a t w gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor1 Real.one_real
                     (List.drop (Arith.nat_of_integer (Z.of_int 2))
                       (Tensor.dims a))
              else l_base_constraint_unsafe (Release (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma)
            (if Arith.less_eq_nat w Arith.one_nata
              then Tensor.tensor0 Real.zero_real (Tensor.dims a)
              else dL_base_constraint_unsafe (Release (c1, c2)) a (Arith.suc t)
                     (Arith.minus_nat w Arith.one_nata) gamma));;

let rec dL_base
  c a t w gamma =
    (if Arith.equal_nata w Arith.zero_nat
      then Tensor.tensor0 Real.zero_real (Tensor.dims a)
      else (if Tensor.valid_index
                 [Arith.minus_nat (Arith.plus_nat t w) Arith.one_nata;
                   max_state_idx_constraint c]
                 (List.take (Arith.nat_of_integer (Z.of_int 2)) (Tensor.dims a))
             then dL_base_constraint_unsafe c a t w gamma
             else failwith "undefined (dL_base invalid index)"));;

end;; (*struct LTLLoss*)

module CodeExport : sig
  val test2 : Arith.int Tensor.tensor
  val test_plus : Arith.int Tensor.tensor
  val test_minus : Arith.int Tensor.tensor
end = struct

let test2 : Arith.int Tensor.tensor
  = Option.the
      (Tensor.subtensor_combine [Arith.nat_of_integer (Z.of_int 2)]
        [Tensor.tensor_from_list
           [Arith.one_int; Arith.Int_of_integer (Z.of_int 2)];
          Tensor.tensor_from_list
            [Arith.Int_of_integer (Z.of_int 3);
              Arith.Int_of_integer (Z.of_int 4)]]);;

let test_plus : Arith.int Tensor.tensor
  = Tensor.plus_tensor Arith.semigroup_add_int test2 test2;;

let test_minus : Arith.int Tensor.tensor
  = Tensor.minus_tensor Arith.minus_int test_plus test2;;

end;; (*struct CodeExport*)
