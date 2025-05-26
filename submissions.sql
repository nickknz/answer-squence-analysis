INSERT INTO y2020_co141_exam.submission
(username, question, part, section, task, ip, answer) VALUES
-- Student 1
('s10000001', 1, 1, 2, 1, '192.168.0.1', 'Yes.'),
('s10000001', 2, 1, 1, 1, '192.168.0.1', 'Yes, that seems right.'),
('s10000001', 2, 1, 1, 2, '192.168.0.1', 'Yes, that seems right based on the lecture notes and the problem context.'),
('s10000001', 2, 1, 2, 1, '192.168.0.1', 'Yes, I agree with the interpretation. It matches our previous discussions and examples.'),
('s10000001', 1, 1, 3, 1, '192.168.0.1', 'Yes, this is the correct answer as it aligns with the definitions and examples we studied in section 2 and 3.'),

-- Student 2
('s10000002', 2, 1, 1, 1, '192.168.0.2', 'No.'),
('s10000002', 1, 1, 2, 1, '192.168.0.2', 'No, I don’t think that is correct.'),
('s10000002', 1, 1, 3, 1, '192.168.0.2', 'No, I don’t think that is correct given the assumptions and constraints.'),
('s10000002', 1, 1, 1, 1, '192.168.0.2', 'No, I disagree with that interpretation based on the information provided and prior knowledge.'),
('s10000002', 2, 1, 2, 1, '192.168.0.2', 'No, that answer seems inconsistent with the methods taught in class and the examples given in the textbook.'),

-- Student 3
('s10000003', 1, 1, 3, 1, '10.0.0.3', 'Maybe.'),
('s10000003', 2, 1, 1, 1, '10.0.0.3', 'Maybe, but I am not entirely certain.'),
('s10000003', 1, 1, 2, 1, '10.0.0.3', 'Maybe, but I am not entirely certain as the question is ambiguous.'),
('s10000003', 2, 1, 2, 1, '10.0.0.3', 'Maybe, although the ambiguity in the wording makes it hard to decide conclusively.'),
('s10000003', 2, 1, 3, 1, '10.0.0.3', 'Maybe, though it would depend on the context and whether the assumption in part b holds for this scenario.'),

-- Student 4
('s10000004', 1, 1, 2, 1, '172.16.1.4', 'I agree.'),
('s10000004', 2, 1, 2, 1, '172.16.1.4', 'I agree with the conclusion based on the result.'),
('s10000004', 2, 1, 3, 1, '172.16.1.4', 'I agree with the conclusion based on the result and how the model behaves in edge cases.'),
('s10000004', 2, 1, 1, 1, '172.16.1.4', 'I agree, particularly because the calculations in section 1 support this as shown in the sample problem.'),
('s10000004', 1, 1, 1, 1, '172.16.1.4', 'I agree with the conclusion given that the derivation in part a and part b match the expected pattern of results and align with section 2.' );
