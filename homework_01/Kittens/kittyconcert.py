from cat import Cat

cat_1 = Cat("Tyrannomeowrus Rex")
cat_2 = Cat("CaptainFluffy")

cat_1.greeting(other = cat_2.name)
cat_2.greeting(other = cat_1.name)