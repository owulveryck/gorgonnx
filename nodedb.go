package main

import "errors"

type nodeDB struct {
	dict map[string]node
}

func (db *nodeDB) addNode(name string, n node) error {
	_, ok := db.dict[name]
	if ok == true {
		return errors.New("Node already exist")
	}
	db.dict[name] = n
	return nil
}

func (db *nodeDB) getNode(name string) (node, error) {
	n, ok := db.dict[name]
	if ok != true {
		return node{}, errors.New("Node does not exist in the database")
	}
	return n, nil
}

func (db *nodeDB) getNodeWithCnx(cnx string) (node, error) {
	return node{}, nil
}
