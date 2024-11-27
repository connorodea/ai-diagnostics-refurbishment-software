--
-- PostgreSQL database dump
--

-- Dumped from database version 14.15 (Homebrew)
-- Dumped by pg_dump version 14.15 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: diagnostics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.diagnostics (
    id integer NOT NULL,
    component_name character varying(100) NOT NULL,
    test_result text NOT NULL,
    test_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    technician_name character varying(100)
);


ALTER TABLE public.diagnostics OWNER TO postgres;

--
-- Name: diagnostics_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.diagnostics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.diagnostics_id_seq OWNER TO postgres;

--
-- Name: diagnostics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.diagnostics_id_seq OWNED BY public.diagnostics.id;


--
-- Name: diagnostics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.diagnostics ALTER COLUMN id SET DEFAULT nextval('public.diagnostics_id_seq'::regclass);


--
-- Data for Name: diagnostics; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.diagnostics (id, component_name, test_result, test_date, technician_name) FROM stdin;
1	CPU	Pass	2024-11-27 01:41:23.445809	John Doe
\.


--
-- Name: diagnostics_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.diagnostics_id_seq', 1, true);


--
-- Name: diagnostics diagnostics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.diagnostics
    ADD CONSTRAINT diagnostics_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

