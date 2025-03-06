(begin

(define (map FUNCTION LIST)
  (
    if (equal? (length LIST) 0)
    () ; true
    (cons (FUNCTION (car LIST)) (map FUNCTION (cdr LIST))) ; false

  )
)

(define (filter FUNCTION LIST)
 (
  if (equal? (length LIST) 0)
    () ; true
    (append
        (
          if (FUNCTION (car LIST))
            (list (car LIST))
            ()
        )
        (filter FUNCTION (cdr LIST))
    )
 )
)


(define (reduce FUNCTION LIST VAL)
 (
  if (equal? (length LIST) 0)
    VAL
    (reduce FUNCTION (cdr LIST) (FUNCTION VAL (car LIST)))
 )
)

)