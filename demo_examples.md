Best demo messages:**

1. **Credit Decision with Precedent Lookup:**
```
Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase.
```
(Jessica Norris has the highest risk score at 0.77, which will trigger precedent searches and policy checks)

2. **Fraud Pattern Analysis:**
```
Analyze the fraud decision history for Jacob Fitzpatrick and find similar cases
```
(High-risk customer with multiple source systems)

3. **Causal Chain Investigation:**
```
What decisions have been made about Amanda Smith and what caused them?
```
(She has the most decisions - 7 total - good for showing decision traces)

4. **Policy-Based Decision:**
```
A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?
```
(Triggers High-Value Transaction Review policy and precedent search)

5. **Exception Request:**
```
We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions.
```
(Shows exception handling with precedent lookup)

**The most compelling single message that demonstrates multiple benefits:**

```
Should we approve a credit limit increase for Jessica Norris? She has a high risk score and is requesting $25,000. Find similar past decisions and applicable policies.
```

This message will:
- Search for the customer (demonstrating entity lookup)
- Find her risk score and decision history
- Search for precedent decisions using semantic/structural similarity
- Look up applicable policies (Credit Limit Policy)
- Show the causal reasoning chain
