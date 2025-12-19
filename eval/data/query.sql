------ Amoxicillin Query ------------------
SELECT generic_name, drug_name, description
FROM boiled
WHERE generic_name LIKE '%amoxicillin%';
-- ORDER BY RANDOM() 
-- LIMIT 110;

------ Adapalene Query --------------------
SELECT generic_name, drug_name, description
FROM boiled
WHERE generic_name LIKE '%adapalene%' ;

------ NSAID Query ------------------------
SELECT generic_name, drug_name, description
FROM boiled
WHERE description LIKE '%nsaid%';
-- ORDER BY RANDOM() 
-- LIMIT 250;

------- Corticosteroid Query --------------
SELECT generic_name, drug_name, description
FROM boiled
WHERE description LIKE '%corticosteroid%';
-- ORDER BY RANDOM() 
-- LIMIT 250;

------- Drugs for "Asthma" Query ----------
SELECT generic_name, drug_name, description
FROM boiled
WHERE description LIKE '%asthma%';
-- ORDER BY RANDOM() 
-- LIMIT 100;