'''
PicoDB simple flat file database
NOTE: fork from TinyDB...

Design:
  API
  * Define a query object similar to TinyDB
  * Define a MaxValue, MinValue object
  Implementation
  * Store one table in a json file.
  * Each row (entry) corresponds to a python dictionary
    * much easier to add/delete fields, query field with 
      missing values, etc.
  * Query of a field introduces a cache of values in table
    (configurable, cache = fast access)
'''
