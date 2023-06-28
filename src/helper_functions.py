def get_emojis_from_member(members, searchID):
  for m in members:
    if m['id'] == searchID:
      return m['emojis']

  return False
